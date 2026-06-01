"""Extra 1: Interpolacion en espacio latente (Seccion 4.4 del paper).

Reproduce las Figs. 8/9 del paper: transiciones suaves entre dos imagenes
codificandolas al espacio ruidoso x_t e interpolando linealmente.

Formula (Ec. latente del paper):
  x_bar_t = (1 - lambda) * x_t + lambda * x_t'
  x_bar_0 ~ p_theta(x_0 | x_bar_t)

No requiere reentrenamiento: usa los pesos DDPM ya existentes.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from ddpm.diffusion import GaussianDiffusion
from ddpm.ddim import DDIMSampler


class LatentSpaceInterpolator:
    """Interpolador en el espacio ruidoso del proceso forward.

    Parameters
    ----------
    diffusion         : instancia de GaussianDiffusion
    interpolation_t   : timestep al que se realiza la interpolacion;
                        t mas alto = mas ruido = interpolacion mas suave
    num_interpolation_steps : puntos de interpolacion entre los dos extremos
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        interpolation_t: int = 500,
        num_interpolation_steps: int = 9,
    ):
        self.diffusion = diffusion
        self.interpolation_t = interpolation_t
        self.num_interpolation_steps = num_interpolation_steps

    def encode_image_to_noisy_latent(
        self,
        x_image: torch.Tensor,
        t: int,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aplica el proceso forward hasta el paso t.

        Usa q_sample (Ec. 4) para obtener x_t desde x_0.
        Fija la semilla para que la interpolacion sea reproducible.

        Returns
        -------
        x_t    : imagen ruidosa en el paso t
        noise  : el ruido gaussiano usado (necesario para comparacion)
        """
        if seed is not None:
            generator = torch.Generator(device=x_image.device).manual_seed(seed)
            noise = torch.randn(x_image.shape, generator=generator, device=x_image.device)
        else:
            noise = torch.randn_like(x_image)

        timesteps_batch = torch.full(
            (x_image.shape[0],), t, device=x_image.device, dtype=torch.long
        )
        x_t = self.diffusion.q_sample(x_image, timesteps_batch, noise)
        return x_t, noise

    def interpolate_noisy_latents(
        self,
        x_t_start: torch.Tensor,
        x_t_end: torch.Tensor,
        interpolation_lambdas: Optional[List[float]] = None,
    ) -> List[torch.Tensor]:
        """Interpolacion lineal entre dos representaciones ruidosas.

        x_bar_t(lambda) = (1 - lambda) * x_t_start + lambda * x_t_end
        lambda=0 -> imagen de inicio, lambda=1 -> imagen final
        """
        if interpolation_lambdas is None:
            interpolation_lambdas = np.linspace(0.0, 1.0, self.num_interpolation_steps).tolist()

        interpolated_latents: List[torch.Tensor] = []
        for lambda_val in interpolation_lambdas:
            x_interpolated = (1.0 - lambda_val) * x_t_start + lambda_val * x_t_end
            interpolated_latents.append(x_interpolated)

        return interpolated_latents

    @torch.no_grad()
    def decode_noisy_latent_to_image(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        device: str = "cuda",
        use_ddim: bool = True,
        ddim_steps: int = 50,
    ) -> torch.Tensor:
        """Decodifica x_t al espacio de imagenes usando el proceso inverso.

        Si use_ddim=True, usa el muestreador rapido DDIM (recomendado para
        generar muchas variantes de interpolacion rapidamente).
        """
        model.eval()

        if use_ddim:
            ddim_sampler = DDIMSampler(self.diffusion, num_steps=ddim_steps, eta=0.0)
            x_decoded = x_t.clone().to(device)
            # Arrancamos desde t=interpolation_t, no desde T
            relevant_steps = [
                i for i, ts in enumerate(ddim_sampler.ddim_timesteps)
                if ts <= self.interpolation_t
            ]
            for step_index in reversed(relevant_steps):
                x_decoded = ddim_sampler.ddim_step(model, x_decoded, step_index)
            return x_decoded
        else:
            x_decoded = x_t.clone().to(device)
            for t in reversed(range(self.interpolation_t)):
                x_decoded = self.diffusion.reverse_step(model, x_decoded, t)
            return x_decoded

    @torch.no_grad()
    def generate_interpolation_grid(
        self,
        model: torch.nn.Module,
        x_image_start: torch.Tensor,
        x_image_end: torch.Tensor,
        noise_seed: int = 42,
        device: str = "cuda",
        ddim_steps: int = 50,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Pipeline completo: encode -> interpola -> decode.

        Genera una secuencia de imagenes que transiciona suavemente
        entre x_image_start y x_image_end.

        Returns
        -------
        decoded_images  : lista de tensores (1, C, H, W) en [-1, 1]
        lambda_values   : lista de valores lambda usados
        """
        model.eval()
        x_image_start = x_image_start.to(device)
        x_image_end = x_image_end.to(device)

        # Codificar ambas imagenes con el mismo ruido base
        x_t_start, _ = self.encode_image_to_noisy_latent(x_image_start, self.interpolation_t, seed=noise_seed)
        x_t_end, _ = self.encode_image_to_noisy_latent(x_image_end, self.interpolation_t, seed=noise_seed + 1)

        lambda_values = np.linspace(0.0, 1.0, self.num_interpolation_steps).tolist()
        interpolated_latents = self.interpolate_noisy_latents(x_t_start, x_t_end, lambda_values)

        decoded_images: List[torch.Tensor] = []
        for x_t_interpolated in interpolated_latents:
            x_decoded = self.decode_noisy_latent_to_image(
                model, x_t_interpolated, device=device,
                use_ddim=True, ddim_steps=ddim_steps,
            )
            decoded_images.append(x_decoded.cpu())

        return decoded_images, lambda_values

    @torch.no_grad()
    def generate_2d_interpolation_grid(
        self,
        model: torch.nn.Module,
        x_image_a: torch.Tensor,
        x_image_b: torch.Tensor,
        x_image_c: torch.Tensor,
        x_image_d: torch.Tensor,
        grid_size: int = 5,
        noise_seed: int = 42,
        device: str = "cuda",
        ddim_steps: int = 50,
    ) -> List[List[torch.Tensor]]:
        """Interpolacion 2D bilineal entre 4 imagenes (esquinas de una cuadricula).

        Genera una cuadricula grid_size x grid_size de imagenes interpoladas.
        Util para visualizar la estructura del espacio latente.
        """
        model.eval()
        images = [img.to(device) for img in [x_image_a, x_image_b, x_image_c, x_image_d]]
        latents = []
        for seed_offset, img in enumerate(images):
            x_t, _ = self.encode_image_to_noisy_latent(img, self.interpolation_t, seed=noise_seed + seed_offset)
            latents.append(x_t)

        x_t_a, x_t_b, x_t_c, x_t_d = latents
        alpha_values = np.linspace(0.0, 1.0, grid_size)
        beta_values = np.linspace(0.0, 1.0, grid_size)

        grid_decoded: List[List[torch.Tensor]] = []
        for alpha in alpha_values:
            row: List[torch.Tensor] = []
            for beta in beta_values:
                x_t_interpolated = (
                    (1 - alpha) * (1 - beta) * x_t_a
                    + (1 - alpha) * beta       * x_t_b
                    + alpha      * (1 - beta)  * x_t_c
                    + alpha      * beta        * x_t_d
                )
                x_decoded = self.decode_noisy_latent_to_image(
                    model, x_t_interpolated, device=device,
                    use_ddim=True, ddim_steps=ddim_steps,
                )
                row.append(x_decoded.cpu())
            grid_decoded.append(row)

        return grid_decoded
