"""Proceso de difusion gaussiana (Ho et al., NeurIPS 2020, arXiv:2006.11239).

Implementa:
  - Schedules de ruido: lineal, coseno, sigmoide
  - GaussianDiffusion: proceso forward (Ec. 4), L_simple (Ec. 14), muestreo ancestral (Alg. 2)
"""
import math
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Schedules de ruido
# ---------------------------------------------------------------------------

def make_linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Schedule lineal del paper original (Seccion 4, DDPM)."""
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def make_cosine_beta_schedule(
    num_timesteps: int,
    offset_s: float = 0.008,
) -> torch.Tensor:
    """Schedule coseno de Nichol & Dhariwal (Improved DDPM, 2021).

    Evita alpha_bar demasiado pequeno cerca de t=0 gracias al offset.
    """
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    f_t = torch.cos(((steps / num_timesteps) + offset_s) / (1.0 + offset_s) * math.pi / 2.0) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-4, max=0.9999).float()


def make_sigmoid_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    sigmoid_range: float = 6.0,
) -> torch.Tensor:
    """Schedule sigmoide: transicion mas suave en los extremos."""
    sigmoid_values = torch.sigmoid(torch.linspace(-sigmoid_range, sigmoid_range, num_timesteps, dtype=torch.float64))
    return (sigmoid_values * (beta_end - beta_start) + beta_start).float()


# ---------------------------------------------------------------------------
# Proceso de difusion gaussiana
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """Proceso de difusion con varianza fija.

    Implementa el paper original Ho et al. 2020:
      - Ec. 4:  q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) I)
      - Ec. 14: L_simple = E[|| eps - eps_theta(x_t, t) ||^2]
      - Alg. 2: muestreo ancestral (reverse process)

    Parameters
    ----------
    betas : torch.Tensor
        Secuencia de varianzas del proceso forward, shape (T,).
    """

    def __init__(self, betas: torch.Tensor):
        self.num_timesteps = int(len(betas))
        betas = betas.double()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Coeficientes del proceso forward
        self._register("betas",                         betas.float())
        self._register("alphas_cumprod",                alphas_cumprod.float())
        self._register("alphas_cumprod_prev",           alphas_cumprod_prev.float())
        self._register("sqrt_alphas_cumprod",           alphas_cumprod.sqrt().float())
        self._register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt().float())
        self._register("sqrt_recip_alphas",             (1.0 / alphas).sqrt().float())

        # Varianza posterior q(x_{t-1} | x_t, x_0)  [Ec. 7]
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self._register("posterior_variance",            posterior_variance.float())
        self._register("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)).float())
        self._register("posterior_mean_coef_x0",
                       (betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)).float())
        self._register("posterior_mean_coef_xt",
                       ((1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)).float())

    def _register(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    def _broadcast_to_batch(self, coeff: torch.Tensor, timesteps: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Extrae coeficientes por timestep y los expande para operar con tensores 4D."""
        values = coeff.to(timesteps.device)[timesteps]
        return values.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))

    # ------------------------------------------------------------------
    # Proceso forward
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ecuacion 4: muestreo directo x_t ~ q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = self._broadcast_to_batch(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha_bar_t = self._broadcast_to_batch(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    # ------------------------------------------------------------------
    # Funcion de perdida
    # ------------------------------------------------------------------

    def compute_loss_simple(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ecuacion 14: L_simple = E[|| eps - eps_theta(x_t, t) ||^2].

        Muestrea t ~ Uniform({1,...,T}) por imagen dentro del batch.
        """
        batch_size = x_start.shape[0]
        if noise is None:
            noise = torch.randn_like(x_start)

        timesteps = torch.randint(
            low=0, high=self.num_timesteps,
            size=(batch_size,), device=x_start.device, dtype=torch.long,
        )
        x_noisy = self.q_sample(x_start, timesteps, noise)
        predicted_noise = model(x_noisy, timesteps)
        return F.mse_loss(predicted_noise, noise)

    # ------------------------------------------------------------------
    # Proceso inverso (muestreo)
    # ------------------------------------------------------------------

    def _predict_x_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruye x_0 a partir de x_t y el ruido predicho."""
        sqrt_recip_alpha_bar_t = self._broadcast_to_batch(
            (1.0 / self.alphas_cumprod).sqrt(), timesteps, x_t.shape)
        sqrt_recip_alpha_bar_minus_one_t = self._broadcast_to_batch(
            (1.0 / self.alphas_cumprod - 1.0).sqrt(), timesteps, x_t.shape)
        return sqrt_recip_alpha_bar_t * x_t - sqrt_recip_alpha_bar_minus_one_t * predicted_noise

    def _compute_posterior_mean(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Media de la posterior q(x_{t-1} | x_t, x_0)  [Ec. 7]."""
        coef_x0 = self._broadcast_to_batch(self.posterior_mean_coef_x0, timesteps, x_t.shape)
        coef_xt = self._broadcast_to_batch(self.posterior_mean_coef_xt, timesteps, x_t.shape)
        return coef_x0 * x_start + coef_xt * x_t

    @torch.no_grad()
    def reverse_step(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Un paso del proceso inverso: x_t -> x_{t-1}  (Algoritmo 2)."""
        batch_size = x_t.shape[0]
        timesteps_batch = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)

        predicted_noise = model(x_t, timesteps_batch)
        x_start_predicted = self._predict_x_start_from_noise(x_t, timesteps_batch, predicted_noise)
        x_start_predicted = x_start_predicted.clamp(-1.0, 1.0)

        posterior_mean = self._compute_posterior_mean(x_start_predicted, x_t, timesteps_batch)

        if t > 0:
            posterior_log_var = self._broadcast_to_batch(
                self.posterior_log_variance_clipped, timesteps_batch, x_t.shape)
            noise = torch.randn_like(x_t)
            x_prev = posterior_mean + (0.5 * posterior_log_var).exp() * noise
        else:
            x_prev = posterior_mean

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        batch_size: int,
        image_channels: int,
        image_size: int,
        device: str = "cuda",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Algoritmo 2: muestreo completo desde ruido puro a imagen.

        Returns tensor (B, C, H, W) en rango [-1, 1].
        """
        shape = (batch_size, image_channels, image_size, image_size)
        x = torch.randn(shape, device=device, generator=generator)

        for t in reversed(range(self.num_timesteps)):
            x = self.reverse_step(model, x, t)

        return x

    @torch.no_grad()
    def sample_progressive(
        self,
        model: torch.nn.Module,
        batch_size: int,
        image_channels: int,
        image_size: int,
        save_at_timesteps: Optional[List[int]] = None,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        """Muestreo con frames intermedios para visualizar la cadena inversa.

        Returns
        -------
        x_final     : tensor final (B, C, H, W)
        saved_frames: lista de (timestep, tensor_cpu) en los pasos indicados
        """
        if save_at_timesteps is None:
            save_at_timesteps = list(range(0, self.num_timesteps, self.num_timesteps // 10))

        shape = (batch_size, image_channels, image_size, image_size)
        x = torch.randn(shape, device=device)
        saved_frames: List[Tuple[int, torch.Tensor]] = [(self.num_timesteps, x.cpu())]

        for t in reversed(range(self.num_timesteps)):
            x = self.reverse_step(model, x, t)
            if t in save_at_timesteps:
                saved_frames.append((t, x.cpu()))

        return x, saved_frames

    # ------------------------------------------------------------------
    # Metricas adicionales
    # ------------------------------------------------------------------

    def compute_vlb_term(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Un termino del VLB en el timestep t (para estimar NLL en bits/dim)."""
        batch_size = x_start.shape[0]
        timesteps_batch = torch.full((batch_size,), t, device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, timesteps_batch, noise)

        with torch.no_grad():
            predicted_noise = model(x_noisy, timesteps_batch)

        x_start_predicted = self._predict_x_start_from_noise(x_noisy, timesteps_batch, predicted_noise)
        x_start_predicted = x_start_predicted.clamp(-1.0, 1.0)

        true_posterior_mean = self._compute_posterior_mean(x_start, x_noisy, timesteps_batch)
        pred_posterior_mean = self._compute_posterior_mean(x_start_predicted, x_noisy, timesteps_batch)

        log_var = self._broadcast_to_batch(self.posterior_log_variance_clipped, timesteps_batch, x_start.shape)
        kl_divergence = 0.5 * (
            (true_posterior_mean - pred_posterior_mean).pow(2) / log_var.exp()
        ).mean(dim=[1, 2, 3])

        return kl_divergence
