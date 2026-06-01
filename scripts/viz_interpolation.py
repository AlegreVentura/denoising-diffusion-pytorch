"""Visualizaciones de interpolacion en espacio latente.

  - plot_linear_interpolation_strip  : tira de imagenes lambda=0..1
  - plot_2d_interpolation_grid       : cuadricula bilineal entre 4 imagenes
  - plot_interpolation_with_latents  : imagenes + representacion del latente
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from .plot_style import (
    apply_dark_style, style_ax,
    BACKGROUND_DARK, AXES_BACKGROUND, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER_COLOR, COLOR_VIOLET, COLOR_BLUE, COLOR_INDIGO,
    tensor_to_uint8_image,
)

apply_dark_style()


def plot_linear_interpolation_strip(
    decoded_images: List["torch.Tensor"],
    lambda_values: List[float],
    interpolation_t: int,
    title: str = "🌊 Interpolación en Espacio Latente",
) -> plt.Figure:
    """Tira horizontal de imagenes interpoladas entre dos puntos.

    La primera imagen es lambda=0 (imagen A) y la ultima lambda=1 (imagen B).
    """
    num_images = len(decoded_images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2.2, 2.8))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    if num_images == 1:
        axes = [axes]

    # Barra de gradiente debajo de las imagenes
    gradient_bar_ax = fig.add_axes([0.1, 0.02, 0.8, 0.03])
    gradient_colors = np.linspace(0, 1, 256)
    gradient_bar_ax.imshow(
        gradient_colors.reshape(1, -1),
        aspect="auto",
        cmap=plt.cm.cool,
        extent=[0, 1, 0, 1],
    )
    gradient_bar_ax.set_xlabel("λ", color=TEXT_SECONDARY, fontsize=9)
    gradient_bar_ax.set_yticks([])
    gradient_bar_ax.tick_params(colors=TEXT_SECONDARY)
    gradient_bar_ax.spines["bottom"].set_color(BORDER_COLOR)
    gradient_bar_ax.set_facecolor(BACKGROUND_DARK)

    for ax, img_tensor, lambda_val in zip(axes, decoded_images, lambda_values):
        ax.set_facecolor(BACKGROUND_DARK)
        if img_tensor.shape[0] == 1:
            img_array = tensor_to_uint8_image(img_tensor[0].repeat(3, 1, 1))
        else:
            img_array = tensor_to_uint8_image(img_tensor[0])
        ax.imshow(img_array)
        ax.set_title(f"λ={lambda_val:.2f}", color=COLOR_VIOLET, fontsize=8.5, fontweight="bold")
        ax.axis("off")

    subtitle = f"Interpolacion en x_t con t={interpolation_t} (Ec. 4 del paper, Sec. 4.4)"
    fig.suptitle(f"{title}\n{subtitle}", color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.08)
    plt.tight_layout(pad=0.2, rect=[0, 0.07, 1, 1])
    return fig


def plot_2d_interpolation_grid(
    grid_decoded: List[List["torch.Tensor"]],
    alpha_values: Optional[List[float]] = None,
    beta_values: Optional[List[float]] = None,
    title: str = "🟦 Interpolación 2D — Cuadrícula Bilineal",
) -> plt.Figure:
    """Cuadricula de imagenes interpoladas bilinealmente entre 4 imagenes."""
    nrows = len(grid_decoded)
    ncols = len(grid_decoded[0]) if nrows > 0 else 0

    if alpha_values is None:
        alpha_values = np.linspace(0, 1, nrows).tolist()
    if beta_values is None:
        beta_values = np.linspace(0, 1, ncols).tolist()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 1.8))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    # Etiquetas de las esquinas
    corner_labels = {
        (0, 0): "A", (0, ncols - 1): "B",
        (nrows - 1, 0): "C", (nrows - 1, ncols - 1): "D",
    }

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor(BACKGROUND_DARK)
            ax.axis("off")

            img_tensor = grid_decoded[row_idx][col_idx]
            if img_tensor.shape[0] == 1:
                img_array = tensor_to_uint8_image(img_tensor[0].repeat(3, 1, 1))
            else:
                img_array = tensor_to_uint8_image(img_tensor[0])
            ax.imshow(img_array)

            if (row_idx, col_idx) in corner_labels:
                ax.set_title(
                    corner_labels[(row_idx, col_idx)],
                    color=COLOR_VIOLET, fontsize=11, fontweight="bold",
                )

    # Etiquetas de ejes
    for col_idx, beta_val in enumerate(beta_values):
        if col_idx % max(1, ncols // 4) == 0:
            axes[0][col_idx].set_title(f"β={beta_val:.1f}", color=TEXT_SECONDARY, fontsize=7)

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.1)
    return fig


def plot_interpolation_t_sensitivity(
    decoded_at_different_t: Dict[int, List["torch.Tensor"]],
    lambda_val: float = 0.5,
    title: str = "🎛️ Sensibilidad al Timestep de Interpolación",
) -> plt.Figure:
    """Muestra como cambia el resultado al interpolar en distintos t.

    t bajo = interpolacion abrupta; t alto = transicion suave.
    """
    t_values = sorted(decoded_at_different_t.keys())
    num_panels = len(t_values)

    fig, axes = plt.subplots(1, num_panels, figsize=(num_panels * 2.2, 2.8))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    if num_panels == 1:
        axes = [axes]

    for ax, t_val in zip(axes, t_values):
        ax.set_facecolor(BACKGROUND_DARK)
        img_list = decoded_at_different_t[t_val]
        mid_idx = len(img_list) // 2
        img_tensor = img_list[mid_idx]

        if img_tensor.shape[0] == 1:
            img_array = tensor_to_uint8_image(img_tensor[0].repeat(3, 1, 1))
        else:
            img_array = tensor_to_uint8_image(img_tensor[0])
        ax.imshow(img_array)
        ax.set_title(f"t_interp={t_val}\nλ={lambda_val:.1f}", color=COLOR_VIOLET, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.2)
    return fig
