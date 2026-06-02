"""DDIM: Denoising Diffusion Implicit Models (Song et al., 2021).

Reutiliza exactamente los pesos de un modelo DDPM entrenado.
Ventaja: muestrea con 10-50x menos pasos sin reentrenar.
  - T=1000 pasos DDPM  -> FID ~10
  - S=100  pasos DDIM  -> FID ~4.2  (10x speedup)
  - S=50   pasos DDIM  -> FID ~4.5
  - S=20   pasos DDIM  -> FID ~6.8

El parametro eta controla el nivel de estocasticidad:
  - eta=0.0 -> completamente deterministico (mismo resultado con misma semilla)
  - eta=1.0 -> equivalente al proceso DDPM original

Referencia: arXiv:2010.02502
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

from .diffusion import GaussianDiffusion


class DDIMSampler:
    """Muestreador DDIM que opera sobre una subsecuencia de timesteps.

    Parameters
    ----------
    diffusion    : GaussianDiffusion con los coeficientes base (T=1000)
    num_steps    : numero de pasos de muestreo DDIM (S << T)
    eta          : nivel de estocasticidad [0, 1]; 0=deterministico
    ddim_discretize: 'uniform' (espaciado uniforme) o 'quad' (cuadratico,
                     mas pasos al inicio donde el ruido es mayor)
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        num_steps: int = 50,
        eta: float = 0.0,
        ddim_discretize: str = "uniform",
    ):
        self.diffusion = diffusion
        self.num_steps = num_steps
        self.eta = eta

        self.ddim_timesteps = self._make_ddim_timestep_sequence(ddim_discretize)
        self._precompute_ddim_coefficients()

    def _make_ddim_timestep_sequence(self, discretize: str) -> np.ndarray:
        """Genera la subsecuencia de timesteps para DDIM."""
        T = self.diffusion.num_timesteps
        if discretize == "uniform":
            step_size = T // self.num_steps
            ddim_timesteps = np.arange(0, T, step_size)
        elif discretize == "quad":
            ddim_timesteps = (np.linspace(0, np.sqrt(T * 0.8), self.num_steps) ** 2).astype(int)
        else:
            raise ValueError(f"ddim_discretize debe ser 'uniform' o 'quad', recibido: {discretize}")
        return ddim_timesteps

    def _precompute_ddim_coefficients(self) -> None:
        """Precalcula alpha_bar en los timesteps DDIM y sus predecesores."""
        alphas_cumprod = self.diffusion.alphas_cumprod.numpy()

        self.ddim_alphas_cumprod = alphas_cumprod[self.ddim_timesteps]
        self.ddim_alphas_cumprod_prev = np.concatenate(
            [alphas_cumprod[:1], alphas_cumprod[self.ddim_timesteps[:-1]]]
        )
        self.ddim_sqrt_one_minus_alphas = np.sqrt(1.0 - self.ddim_alphas_cumprod)

        # sigma_t = eta * sqrt( (1-alpha_prev)/(1-alpha_t) * (1 - alpha_t/alpha_prev) )
        # Ec. 16 de Song et al. (2021). El producto bajo la raiz es siempre >= 0
        # porque alpha_prev >= alpha_t (mayor t = menor alpha_bar).
        self.ddim_sigmas = self.eta * np.sqrt(
            np.maximum(0.0,
                (1.0 - self.ddim_alphas_cumprod_prev)
                / np.maximum(1.0 - self.ddim_alphas_cumprod, 1e-8)
                * (1.0 - self.ddim_alphas_cumprod / np.maximum(self.ddim_alphas_cumprod_prev, 1e-8))
            )
        )

    @torch.no_grad()
    def ddim_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """Un paso DDIM: x_{t_i} -> x_{t_{i-1}}.

        Formula DDIM (Ec. 12 de Song et al.):
          x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred
                  + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps_theta
                  + sigma_t * z
        """
        batch_size = x_t.shape[0]
        t_current = self.ddim_timesteps[step_index]
        timesteps_batch = torch.full((batch_size,), t_current, device=x_t.device, dtype=torch.long)

        # Predecir el ruido con el modelo DDPM (sin cambios en el modelo)
        predicted_noise = model(x_t, timesteps_batch)

        # Coeficientes para este paso
        alpha_bar_t = self.ddim_alphas_cumprod[step_index]
        alpha_bar_t_prev = self.ddim_alphas_cumprod_prev[step_index]
        sigma_t = self.ddim_sigmas[step_index]
        sqrt_one_minus_alpha_t = self.ddim_sqrt_one_minus_alphas[step_index]

        # Reconstruir x_0 predicho
        x0_predicted = (
            x_t.cpu().numpy() / np.sqrt(alpha_bar_t)
            - predicted_noise.cpu().numpy() * sqrt_one_minus_alpha_t / np.sqrt(alpha_bar_t)
        )
        x0_predicted = np.clip(x0_predicted, -1.0, 1.0)
        x0_predicted_tensor = torch.from_numpy(x0_predicted).to(x_t.device)

        # Componente de "direccion hacia x_t" (clamped a 0 para evitar sqrt negativo)
        direction_coeff = np.maximum(0.0, 1.0 - alpha_bar_t_prev - sigma_t ** 2)
        direction_pointing_to_xt = np.sqrt(direction_coeff) * predicted_noise.cpu().numpy()
        direction_tensor = torch.from_numpy(direction_pointing_to_xt).to(x_t.device)

        # Componente deterministico
        x_prev = (
            np.sqrt(alpha_bar_t_prev) * x0_predicted
            + direction_pointing_to_xt
        )
        x_prev_tensor = torch.from_numpy(x_prev).to(x_t.device).float()

        # Componente estocastico (solo si eta > 0)
        if sigma_t > 0:
            noise = torch.randn_like(x_t)
            x_prev_tensor = x_prev_tensor + sigma_t * noise

        return x_prev_tensor

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        image_channels: int,
        image_size: int,
        device: str = "cuda",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Genera imagenes con muestreo DDIM.

        Returns tensor (B, C, H, W) en rango [-1, 1].
        """
        shape = (batch_size, image_channels, image_size, image_size)
        x = torch.randn(shape, device=device, generator=generator)

        for step_index in reversed(range(len(self.ddim_timesteps))):
            x = self.ddim_step(model, x, step_index)

        return x

    @torch.no_grad()
    def sample_progressive(
        self,
        model: nn.Module,
        batch_size: int,
        image_channels: int,
        image_size: int,
        save_every_n_steps: int = 5,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        """Muestreo DDIM con frames intermedios para visualizacion."""
        shape = (batch_size, image_channels, image_size, image_size)
        x = torch.randn(shape, device=device)
        saved_frames: List[Tuple[int, torch.Tensor]] = [(self.num_steps, x.cpu())]

        for step_index in reversed(range(len(self.ddim_timesteps))):
            x = self.ddim_step(model, x, step_index)
            reverse_step_index = len(self.ddim_timesteps) - 1 - step_index
            if reverse_step_index % save_every_n_steps == 0:
                saved_frames.append((self.ddim_timesteps[step_index], x.cpu()))

        return x, saved_frames

    def benchmark_sampling_time(
        self,
        model: nn.Module,
        batch_size: int,
        image_channels: int,
        image_size: int,
        num_warmup_runs: int = 2,
        num_timed_runs: int = 5,
        device: str = "cuda",
    ) -> dict:
        """Mide el tiempo de muestreo para comparacion DDPM vs DDIM."""
        import time

        for _ in range(num_warmup_runs):
            self.sample(model, batch_size, image_channels, image_size, device)
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(num_timed_runs):
            start = time.perf_counter()
            self.sample(model, batch_size, image_channels, image_size, device)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        return {
            "num_steps": self.num_steps,
            "eta": self.eta,
            "mean_seconds": float(np.mean(times)),
            "std_seconds": float(np.std(times)),
            "seconds_per_image": float(np.mean(times)) / batch_size,
        }
