"""Visualizaciones del proceso de difusion (forward y reverse).

Funciones para el notebook showcase:
  - plot_forward_process_strip   : cadena x_0 -> x_T para una imagen
  - plot_reverse_process_strip   : cadena x_T -> x_0 (muestreo)
  - plot_noise_schedule_overview : beta_t, alpha_bar_t, SNR para los 3 schedules
  - plot_image_grid              : cuadricula generica de imagenes
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Optional, Dict

from .plot_style import (
    apply_dark_style, make_figure, make_grid_figure, style_ax,
    BACKGROUND_DARK, AXES_BACKGROUND, TEXT_PRIMARY, TEXT_SECONDARY,
    TEXT_MUTED, BORDER_COLOR, COLOR_INDIGO, COLOR_BLUE, COLOR_EMERALD,
    SCHEDULE_COLORS, tensor_to_uint8_image,
)

apply_dark_style()


def plot_forward_process_strip(
    x_clean: "torch.Tensor",
    noisy_samples: Dict[int, "torch.Tensor"],
    title: str = "⬛ → ☁️  Proceso Forward: destruccion progresiva",
) -> plt.Figure:
    """Tira horizontal: imagen limpia y versiones ruidosas en distintos t.

    Parameters
    ----------
    x_clean       : tensor (C, H, W) de la imagen original
    noisy_samples : dict {t: tensor (C, H, W)} con x_t en varios pasos
    """
    timesteps_sorted = sorted(noisy_samples.keys())
    num_panels = 1 + len(timesteps_sorted)

    fig, axes = plt.subplots(1, num_panels, figsize=(num_panels * 2.2, 2.8))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    # Imagen original
    axes[0].imshow(tensor_to_uint8_image(x_clean), cmap=None if x_clean.shape[0] == 3 else "gray")
    axes[0].set_title("x₀\noriginal", color=TEXT_PRIMARY, fontsize=9, fontweight="bold")
    axes[0].axis("off")

    # Versiones ruidosas
    for ax, t in zip(axes[1:], timesteps_sorted):
        img = noisy_samples[t]
        if img.shape[0] == 1:
            axes_img = tensor_to_uint8_image(img.repeat(3, 1, 1))
        else:
            axes_img = tensor_to_uint8_image(img)
        ax.imshow(axes_img)
        ax.set_title(f"x_t\nt={t}", color=TEXT_SECONDARY, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.3)
    return fig


def plot_reverse_process_strip(
    saved_frames: List[Tuple[int, "torch.Tensor"]],
    image_idx: int = 0,
    title: str = "☁️ → 🖼️  Proceso Inverso: emergencia desde ruido puro",
) -> plt.Figure:
    """Tira horizontal de frames del proceso de muestreo (reverse).

    Parameters
    ----------
    saved_frames : lista de (timestep, batch_tensor) guardados durante el muestreo
    image_idx    : cual imagen del batch mostrar
    """
    frames_sorted = sorted(saved_frames, key=lambda x: x[0], reverse=True)
    num_panels = len(frames_sorted)

    fig, axes = plt.subplots(1, num_panels, figsize=(num_panels * 2.0, 2.8))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    if num_panels == 1:
        axes = [axes]

    for ax, (t, batch_tensor) in zip(axes, frames_sorted):
        img_tensor = batch_tensor[image_idx]
        if img_tensor.shape[0] == 1:
            img_array = tensor_to_uint8_image(img_tensor.repeat(3, 1, 1))
        else:
            img_array = tensor_to_uint8_image(img_tensor)
        ax.imshow(img_array)
        ax.set_title(f"t={t}", color=TEXT_SECONDARY, fontsize=8)
        ax.axis("off")

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.3)
    return fig


def plot_noise_schedule_overview(
    schedules_data: Dict[str, Dict],
    title: str = "📊 Comparativa de Schedules de Ruido",
) -> plt.Figure:
    """Panel 2x2 con beta_t, alpha_bar_t, SNR y zoom del SNR en escala log.

    Parameters
    ----------
    schedules_data : dict {nombre: {'timesteps': np.array, 'betas': np.array,
                                    'alphas_cumprod': np.array, 'snr': np.array}}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    panel_specs = [
        ("betas",         "beta_t",          "Varianza del ruido añadido",       "linear"),
        ("alphas_cumprod","alpha_bar_t",      "Señal restante en x_t",            "linear"),
        ("snr",           "SNR(t)",           "Signal-to-Noise Ratio",            "log"),
    ]

    for ax, (data_key, ylabel, panel_title, yscale) in zip(axes, panel_specs):
        for schedule_name, data in schedules_data.items():
            label = {
                "linear":  "Lineal (DDPM)",
                "cosine":  "Coseno (iDDPM)",
                "sigmoid": "Sigmoide",
            }.get(schedule_name, schedule_name)
            ax.plot(
                data["timesteps"], data[data_key],
                color=SCHEDULE_COLORS.get(schedule_name, COLOR_INDIGO),
                label=label, linewidth=2.2,
            )
        style_ax(ax, xlabel="Timestep t", ylabel=ylabel, title=panel_title, yscale=yscale)
        ax.legend(loc="best")
        ax.set_facecolor(AXES_BACKGROUND)

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.0)
    return fig


def plot_image_grid(
    image_tensors: List["torch.Tensor"],
    nrows: int = 4,
    ncols: int = 8,
    title: str = "Imagenes generadas",
    caption_texts: Optional[List[str]] = None,
) -> plt.Figure:
    """Cuadricula de imagenes con fondo oscuro."""
    num_images = min(len(image_tensors), nrows * ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    fig.patch.set_facecolor(BACKGROUND_DARK)
    axes_flat = axes.flatten()

    for idx in range(nrows * ncols):
        ax = axes_flat[idx]
        ax.set_facecolor(BACKGROUND_DARK)
        if idx < num_images:
            img_tensor = image_tensors[idx]
            if img_tensor.shape[0] == 1:
                img_array = tensor_to_uint8_image(img_tensor.repeat(3, 1, 1))
            else:
                img_array = tensor_to_uint8_image(img_tensor)
            ax.imshow(img_array)
            if caption_texts and idx < len(caption_texts):
                ax.set_title(caption_texts[idx], color=TEXT_SECONDARY, fontsize=7, pad=2)
        ax.axis("off")

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.1)
    return fig
