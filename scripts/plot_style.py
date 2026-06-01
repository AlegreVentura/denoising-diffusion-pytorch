"""Estilo visual unificado para todas las graficas del proyecto DDPM.

Tema oscuro formal: fondo #0d1117, acentos indigo/azul/verde.
Importar este modulo antes de cualquier llamada a matplotlib.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Paleta de colores del proyecto
# ---------------------------------------------------------------------------

BACKGROUND_DARK     = "#0d1117"
AXES_BACKGROUND     = "#161b22"
GRID_COLOR          = "#21262d"
BORDER_COLOR        = "#30363d"
TEXT_PRIMARY        = "#f0f6fc"
TEXT_SECONDARY      = "#8b949e"
TEXT_MUTED          = "#484f58"

COLOR_INDIGO        = "#818cf8"   # schedule lineal / epsilon
COLOR_BLUE          = "#38bdf8"   # schedule coseno / DDIM
COLOR_EMERALD       = "#34d399"   # schedule sigmoide / v-prediction
COLOR_AMBER         = "#fbbf24"   # metricas / advertencias
COLOR_ROSE          = "#fb7185"   # perdidas / errores
COLOR_VIOLET        = "#c084fc"   # interpolacion

SCHEDULE_COLORS = {
    "linear":  COLOR_INDIGO,
    "cosine":  COLOR_BLUE,
    "sigmoid": COLOR_EMERALD,
}
PREDICTION_COLORS = {
    "epsilon":  COLOR_INDIGO,
    "x0":       COLOR_AMBER,
    "v":        COLOR_EMERALD,
}
SAMPLER_COLORS = {
    "DDPM":  COLOR_ROSE,
    "DDIM":  COLOR_BLUE,
}


# ---------------------------------------------------------------------------
# Aplicar estilo global
# ---------------------------------------------------------------------------

def apply_dark_style() -> None:
    """Configura matplotlib con el tema oscuro del proyecto."""
    mpl.rcParams.update({
        # Figura
        "figure.facecolor":     BACKGROUND_DARK,
        "figure.edgecolor":     BACKGROUND_DARK,
        "figure.dpi":           130,

        # Ejes
        "axes.facecolor":       AXES_BACKGROUND,
        "axes.edgecolor":       BORDER_COLOR,
        "axes.labelcolor":      TEXT_SECONDARY,
        "axes.titlecolor":      TEXT_PRIMARY,
        "axes.titlesize":       13,
        "axes.labelsize":       11,
        "axes.grid":            True,
        "axes.spines.top":      False,
        "axes.spines.right":    False,

        # Grid
        "grid.color":           GRID_COLOR,
        "grid.linewidth":       0.7,
        "grid.alpha":           1.0,

        # Texto
        "text.color":           TEXT_PRIMARY,
        "font.family":          "DejaVu Sans",
        "font.size":            10,

        # Ticks
        "xtick.color":          TEXT_SECONDARY,
        "ytick.color":          TEXT_SECONDARY,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,

        # Leyenda
        "legend.facecolor":     AXES_BACKGROUND,
        "legend.edgecolor":     BORDER_COLOR,
        "legend.labelcolor":    TEXT_PRIMARY,
        "legend.fontsize":      9,
        "legend.framealpha":    0.95,

        # Lineas
        "lines.linewidth":      2.0,
        "lines.antialiased":    True,

        # Imagen
        "image.cmap":           "gray",

        # Savefig
        "savefig.facecolor":    BACKGROUND_DARK,
        "savefig.edgecolor":    BACKGROUND_DARK,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.15,
    })


# Aplicar al importar
apply_dark_style()


# ---------------------------------------------------------------------------
# Helpers de layout
# ---------------------------------------------------------------------------

def make_figure(
    width: float = 10.0,
    height: float = 5.0,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, fontsize=14, fontweight="bold", y=1.01)
    return fig, ax


def make_grid_figure(
    nrows: int,
    ncols: int,
    width_per_col: float = 3.5,
    height_per_row: float = 3.0,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * width_per_col, nrows * height_per_row),
    )
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, fontsize=14, fontweight="bold", y=1.01)
    return fig, axes


def style_ax(
    ax: plt.Axes,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    xscale: str = "linear",
    yscale: str = "linear",
) -> None:
    """Aplica formato estandar a un eje."""
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontweight="bold", pad=8)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)


def add_accent_line(ax: plt.Axes, value: float, orientation: str = "h", label: str = "", color: str = COLOR_AMBER) -> None:
    """Linea punteada de referencia (p.ej. FID objetivo, umbral)."""
    if orientation == "h":
        ax.axhline(value, color=color, linestyle="--", linewidth=1.2, alpha=0.7, label=label)
    else:
        ax.axvline(value, color=color, linestyle="--", linewidth=1.2, alpha=0.7, label=label)


def tensor_to_uint8_image(tensor_image: "torch.Tensor") -> np.ndarray:
    """Convierte tensor (C,H,W) en [-1,1] a array (H,W,C) uint8."""
    img = tensor_image.detach().cpu().float()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img
