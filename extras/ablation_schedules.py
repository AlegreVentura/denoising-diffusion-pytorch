"""Extra 3: Ablacion sistematica de schedules de ruido.

Compara lineal vs coseno vs sigmoide en terminos de:
  - Curva SNR (Signal-to-Noise Ratio) a lo largo del proceso
  - alpha_bar_t (cuanto de la imagen original queda en x_t)
  - Velocidad de destruccion de la informacion
  - Impacto en el entrenamiento (gradientes, perdida)

Referencia: Nichol & Dhariwal, Improved DDPM (2021), Fig. 5 y 6.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple

from ddpm.diffusion import (
    GaussianDiffusion,
    make_linear_beta_schedule,
    make_cosine_beta_schedule,
    make_sigmoid_beta_schedule,
)


SCHEDULE_NAMES = ("linear", "cosine", "sigmoid")
SCHEDULE_LABELS = {
    "linear":  "Lineal (DDPM original)",
    "cosine":  "Coseno (Improved DDPM)",
    "sigmoid": "Sigmoide",
}


class NoiseScheduleAblation:
    """Analiza y compara los tres schedules de ruido del proyecto.

    Parameters
    ----------
    num_timesteps : longitud del schedule (T=1000 en el paper)
    """

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        self.diffusion_objects: Dict[str, GaussianDiffusion] = self._build_all_diffusions()

    def _build_all_diffusions(self) -> Dict[str, GaussianDiffusion]:
        betas = {
            "linear":  make_linear_beta_schedule(self.num_timesteps),
            "cosine":  make_cosine_beta_schedule(self.num_timesteps),
            "sigmoid": make_sigmoid_beta_schedule(self.num_timesteps),
        }
        return {name: GaussianDiffusion(beta_schedule) for name, beta_schedule in betas.items()}

    # ------------------------------------------------------------------
    # Metricas de analisis del schedule
    # ------------------------------------------------------------------

    def compute_snr_curve(self, schedule_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """SNR(t) = alpha_bar_t / (1 - alpha_bar_t).

        SNR alto = poca degradacion; SNR ~0 = imagen destruida.
        El paper usa SNR para pesar la perdida (aunque con L_simple se ignora el peso).
        """
        diffusion = self.diffusion_objects[schedule_name]
        alpha_bar = diffusion.alphas_cumprod.numpy()
        snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
        timesteps = np.arange(self.num_timesteps)
        return timesteps, snr

    def compute_alpha_bar_curve(self, schedule_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """alpha_bar_t: fraccion de senal original que queda en x_t."""
        diffusion = self.diffusion_objects[schedule_name]
        alpha_bar = diffusion.alphas_cumprod.numpy()
        timesteps = np.arange(self.num_timesteps)
        return timesteps, alpha_bar

    def compute_beta_curve(self, schedule_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """beta_t: varianza anadida en cada paso."""
        diffusion = self.diffusion_objects[schedule_name]
        betas = diffusion.betas.numpy()
        timesteps = np.arange(self.num_timesteps)
        return timesteps, betas

    def find_half_signal_timestep(self, schedule_name: str) -> int:
        """Timestep donde alpha_bar_t = 0.5 (mitad de la senal original)."""
        diffusion = self.diffusion_objects[schedule_name]
        alpha_bar = diffusion.alphas_cumprod.numpy()
        half_signal_idx = np.searchsorted(-alpha_bar, -0.5)
        return int(half_signal_idx)

    def compute_all_metrics_summary(self) -> Dict[str, Dict]:
        """Tabla resumen de estadisticas clave por schedule."""
        summary: Dict[str, Dict] = {}
        for name in SCHEDULE_NAMES:
            diffusion = self.diffusion_objects[name]
            alpha_bar = diffusion.alphas_cumprod.numpy()
            betas = diffusion.betas.numpy()

            summary[name] = {
                "label": SCHEDULE_LABELS[name],
                "beta_min": float(betas.min()),
                "beta_max": float(betas.max()),
                "alpha_bar_at_T_half": float(alpha_bar[self.num_timesteps // 2]),
                "alpha_bar_at_T": float(alpha_bar[-1]),
                "half_signal_timestep": self.find_half_signal_timestep(name),
                "snr_at_T_half": float(alpha_bar[self.num_timesteps // 2] /
                                        (1.0 - alpha_bar[self.num_timesteps // 2] + 1e-8)),
            }
        return summary

    # ------------------------------------------------------------------
    # Visualizacion del proceso forward con distintos schedules
    # ------------------------------------------------------------------

    def apply_forward_process_comparison(
        self,
        x_clean: torch.Tensor,
        timesteps_to_show: List[int],
        noise_seed: int = 0,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """Aplica q_sample en los timesteps indicados para cada schedule.

        Devuelve un dict {schedule_name: {t: x_t}} para visualizar como
        el mismo ruido destruye la imagen a diferente velocidad segun el schedule.
        """
        generator = torch.Generator().manual_seed(noise_seed)
        noise = torch.randn(x_clean.shape, generator=generator)

        forward_samples: Dict[str, Dict[int, torch.Tensor]] = {}
        for schedule_name, diffusion in self.diffusion_objects.items():
            forward_samples[schedule_name] = {}
            for t in timesteps_to_show:
                if t >= self.num_timesteps:
                    continue
                timesteps_batch = torch.tensor([t], dtype=torch.long)
                x_t = diffusion.q_sample(
                    x_clean[:1], timesteps_batch, noise[:1]
                )
                forward_samples[schedule_name][t] = x_t.detach().cpu()

        return forward_samples

    # ------------------------------------------------------------------
    # Entrenamiento comparativo (loss por schedule)
    # ------------------------------------------------------------------

    def compute_loss_landscape(
        self,
        model: torch.nn.Module,
        x_batch: torch.Tensor,
        num_timestep_samples: int = 50,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Calcula L_simple en funcion de t para cada schedule.

        Util para ver si algunos schedules producen gradientes mas estables.
        """
        import torch.nn.functional as F

        model.eval()
        device = next(model.parameters()).device
        x_batch = x_batch.to(device)

        results: Dict[str, Dict[str, np.ndarray]] = {}
        sampled_timesteps = np.linspace(0, self.num_timesteps - 1, num_timestep_samples, dtype=int)

        with torch.no_grad():
            for schedule_name, diffusion in self.diffusion_objects.items():
                losses_at_t = []
                for t in sampled_timesteps:
                    timesteps_batch = torch.full(
                        (x_batch.shape[0],), t, device=device, dtype=torch.long
                    )
                    noise = torch.randn_like(x_batch)
                    x_noisy = diffusion.q_sample(x_batch, timesteps_batch, noise)
                    predicted_noise = model(x_noisy, timesteps_batch)
                    loss = F.mse_loss(predicted_noise, noise).item()
                    losses_at_t.append(loss)

                results[schedule_name] = {
                    "timesteps": sampled_timesteps.astype(float),
                    "loss": np.array(losses_at_t),
                }

        return results
