"""Extra 3b: Parametrizacion v (Salimans & Ho, 2022, arXiv:2202.00512).

Tercera parametrizacion para la red:
  - eps-prediction  (DDPM original): predice el ruido eps
  - x0-prediction:  predice la imagen limpia x_0
  - v-prediction:   predice v = sqrt(alpha_bar_t)*eps - sqrt(1-alpha_bar_t)*x_0

La v-prediction es mas estable para pocos pasos de muestreo y fue
adoptada por Stable Diffusion v2 y Vision Transformers de difusion.

Esta clase extiende GaussianDiffusion con la nueva parametrizacion,
permitiendo una ablacion directa: epsilon vs x0 vs v.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ddpm.diffusion import GaussianDiffusion


PREDICTION_TYPES = ("epsilon", "x0", "v")


class VPredictionDiffusion(GaussianDiffusion):
    """GaussianDiffusion con soporte para las tres parametrizaciones.

    Parameters
    ----------
    betas           : schedule de ruido
    prediction_type : 'epsilon' (default DDPM), 'x0', o 'v'
    """

    def __init__(self, betas: torch.Tensor, prediction_type: str = "epsilon"):
        super().__init__(betas)
        assert prediction_type in PREDICTION_TYPES, (
            f"prediction_type debe ser uno de {PREDICTION_TYPES}"
        )
        self.prediction_type = prediction_type

        # Coeficientes adicionales para v-prediction
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        self._register("sqrt_alphas_cumprod_float64",        sqrt_alphas_cumprod)
        self._register("sqrt_one_minus_alphas_cumprod_float64", sqrt_one_minus_alphas_cumprod)

    # ------------------------------------------------------------------
    # Funciones de conversion entre parametrizaciones
    # ------------------------------------------------------------------

    def compute_v_from_epsilon_and_x0(
        self,
        epsilon: torch.Tensor,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """v_t = sqrt(alpha_bar_t)*eps - sqrt(1-alpha_bar_t)*x_0."""
        sqrt_alpha_bar_t = self._broadcast_to_batch(self.sqrt_alphas_cumprod, timesteps, epsilon.shape)
        sqrt_one_minus_alpha_bar_t = self._broadcast_to_batch(self.sqrt_one_minus_alphas_cumprod, timesteps, epsilon.shape)
        return sqrt_alpha_bar_t * epsilon - sqrt_one_minus_alpha_bar_t * x_start

    def predict_epsilon_from_v(
        self,
        v_predicted: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Convierte prediccion v -> epsilon predicho.

        eps = sqrt(alpha_bar_t)*v + sqrt(1-alpha_bar_t)*x_t
        """
        sqrt_alpha_bar_t = self._broadcast_to_batch(self.sqrt_alphas_cumprod, timesteps, v_predicted.shape)
        sqrt_one_minus_alpha_bar_t = self._broadcast_to_batch(self.sqrt_one_minus_alphas_cumprod, timesteps, v_predicted.shape)
        return sqrt_alpha_bar_t * v_predicted + sqrt_one_minus_alpha_bar_t * x_t

    def predict_x0_from_v(
        self,
        v_predicted: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Convierte prediccion v -> x_0 predicho.

        x_0 = sqrt(alpha_bar_t)*x_t - sqrt(1-alpha_bar_t)*v
        """
        sqrt_alpha_bar_t = self._broadcast_to_batch(self.sqrt_alphas_cumprod, timesteps, v_predicted.shape)
        sqrt_one_minus_alpha_bar_t = self._broadcast_to_batch(self.sqrt_one_minus_alphas_cumprod, timesteps, v_predicted.shape)
        return sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * v_predicted

    def predict_x0_from_epsilon(
        self,
        epsilon_predicted: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Convierte prediccion epsilon -> x_0 predicho."""
        sqrt_recip_alpha_bar_t = self._broadcast_to_batch(
            (1.0 / self.alphas_cumprod).sqrt(), timesteps, epsilon_predicted.shape
        )
        sqrt_recip_alpha_bar_minus_one_t = self._broadcast_to_batch(
            (1.0 / self.alphas_cumprod - 1.0).sqrt(), timesteps, epsilon_predicted.shape
        )
        return sqrt_recip_alpha_bar_t * x_t - sqrt_recip_alpha_bar_minus_one_t * epsilon_predicted

    # ------------------------------------------------------------------
    # Loss unificada para las tres parametrizaciones
    # ------------------------------------------------------------------

    def compute_loss_with_prediction_type(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """L_simple adaptada segun self.prediction_type.

        - epsilon: MSE(eps, model(x_t, t))          [DDPM original]
        - x0:      MSE(x_0, model(x_t, t))
        - v:       MSE(v_t, model(x_t, t))
        """
        batch_size = x_start.shape[0]
        if noise is None:
            noise = torch.randn_like(x_start)

        timesteps = torch.randint(
            low=0, high=self.num_timesteps,
            size=(batch_size,), device=x_start.device, dtype=torch.long,
        )
        x_noisy = self.q_sample(x_start, timesteps, noise)
        model_output = model(x_noisy, timesteps)

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = x_start
        elif self.prediction_type == "v":
            target = self.compute_v_from_epsilon_and_x0(noise, x_start, timesteps)

        return F.mse_loss(model_output, target)

    # ------------------------------------------------------------------
    # Muestreo adaptado para las tres parametrizaciones
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reverse_step_unified(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Un paso inverso compatible con las tres parametrizaciones."""
        batch_size = x_t.shape[0]
        timesteps_batch = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)

        model_output = model(x_t, timesteps_batch)

        # Convertir la prediccion del modelo a x_0 estimado
        if self.prediction_type == "epsilon":
            x0_predicted = self._predict_x_start_from_noise(x_t, timesteps_batch, model_output)
        elif self.prediction_type == "x0":
            x0_predicted = model_output
        elif self.prediction_type == "v":
            x0_predicted = self.predict_x0_from_v(model_output, x_t, timesteps_batch)

        x0_predicted = x0_predicted.clamp(-1.0, 1.0)
        posterior_mean = self._compute_posterior_mean(x0_predicted, x_t, timesteps_batch)

        if t > 0:
            posterior_log_var = self._broadcast_to_batch(
                self.posterior_log_variance_clipped, timesteps_batch, x_t.shape
            )
            noise = torch.randn_like(x_t)
            x_prev = posterior_mean + (0.5 * posterior_log_var).exp() * noise
        else:
            x_prev = posterior_mean

        return x_prev

    @torch.no_grad()
    def sample_unified(
        self,
        model: nn.Module,
        batch_size: int,
        image_channels: int,
        image_size: int,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Muestreo completo compatible con las tres parametrizaciones."""
        shape = (batch_size, image_channels, image_size, image_size)
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            x = self.reverse_step_unified(model, x, t)
        return x
