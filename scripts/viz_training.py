"""Visualizaciones del proceso de entrenamiento.

  - plot_training_loss_curves        : train/val loss a lo largo del tiempo
  - plot_gradient_norms              : normas L2 por capa en escala log
  - plot_gradient_flow_heatmap       : mapa de calor de normas por capa/paso
  - plot_ema_vs_raw_comparison       : FID/loss con y sin EMA
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple

from .plot_style import (
    apply_dark_style, style_ax, add_accent_line,
    BACKGROUND_DARK, AXES_BACKGROUND, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER_COLOR, COLOR_INDIGO, COLOR_BLUE, COLOR_EMERALD,
    COLOR_AMBER, COLOR_ROSE, COLOR_VIOLET,
)

apply_dark_style()


def plot_training_loss_curves(
    train_steps: List[int],
    train_losses: List[float],
    val_steps: Optional[List[int]] = None,
    val_losses: Optional[List[float]] = None,
    smoothing_window: int = 50,
    title: str = "Curvas de Perdida — Entrenamiento vs Validacion",
    warmup_steps_to_skip: int = 500,
) -> plt.Figure:
    """Panel izquierdo: vista completa. Panel derecho: zoom post-warmup.

    El spike inicial del warmup domina el eje Y en la vista completa,
    por eso el panel derecho muestra la zona de convergencia real.
    """
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    train_arr  = np.array(train_losses)
    train_steps_arr = np.array(train_steps)

    # Suavizado
    smoothed_train = train_arr
    smoothed_steps = train_steps_arr
    if len(train_arr) >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed_train = np.convolve(train_arr, kernel, mode="valid")
        offset = smoothing_window // 2
        smoothed_steps = train_steps_arr[offset: offset + len(smoothed_train)]

    val_arr   = np.array(val_losses)   if val_losses else np.array([])
    val_steps_arr = np.array(val_steps) if val_steps  else np.array([])

    for ax, panel_title in [(ax_full, "Vista completa"), (ax_zoom, f"Zoom (paso > {warmup_steps_to_skip})")]:
        ax.set_facecolor(AXES_BACKGROUND)

        if ax is ax_zoom:
            mask_t = smoothed_steps > warmup_steps_to_skip
            mask_r = train_steps_arr > warmup_steps_to_skip
            plot_steps   = smoothed_steps[mask_t]
            plot_smooth  = smoothed_train[mask_t]
            plot_raw_s   = train_steps_arr[mask_r]
            plot_raw     = train_arr[mask_r]
        else:
            plot_steps  = smoothed_steps
            plot_smooth = smoothed_train
            plot_raw_s  = train_steps_arr
            plot_raw    = train_arr

        # Loss cruda muy suave de fondo
        ax.plot(plot_raw_s, plot_raw, color=COLOR_INDIGO, alpha=0.12, linewidth=0.6)
        ax.plot(plot_steps, plot_smooth, color=COLOR_INDIGO, linewidth=2.2,
                label=f"Train (suavizado {smoothing_window})")

        if len(val_arr) > 0:
            if ax is ax_zoom:
                mask_v = val_steps_arr > warmup_steps_to_skip
                v_s = val_steps_arr[mask_v]
                v_l = val_arr[mask_v]
            else:
                v_s, v_l = val_steps_arr, val_arr

            ax.plot(v_s, v_l, color=COLOR_AMBER, linewidth=2.0,
                    marker="o", markersize=3.5,
                    markevery=max(1, len(v_s) // 25),
                    label="Validacion loss")

        # Anotacion del valor final
        if len(plot_smooth) > 0:
            final_loss = plot_smooth[-1]
            final_step = plot_steps[-1]
            ax.annotate(
                f" {final_loss:.4f}",
                xy=(final_step, final_loss),
                color=COLOR_INDIGO, fontsize=8.5,
                va="center",
            )

        style_ax(ax, xlabel="Paso de entrenamiento",
                 ylabel="L_simple (MSE)" if ax is ax_full else "",
                 title=panel_title)
        ax.legend(loc="upper right", fontsize=8)

        # En el zoom, acotar el eje Y al rango real de convergencia
        if ax is ax_zoom and len(plot_smooth) > 0:
            p5  = np.percentile(plot_smooth, 2)
            p95 = np.percentile(plot_smooth, 98)
            margin = (p95 - p5) * 0.3
            ax.set_ylim(max(0, p5 - margin), p95 + margin)

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=1.2)
    return fig


def plot_gradient_norms(
    steps: List[int],
    global_norms: List[float],
    layer_norms_history: Optional[Dict[str, List[float]]] = None,
    max_layers_to_show: int = 8,
    title: str = "🔬 Normas L2 de Gradientes por Capa",
) -> plt.Figure:
    """Plot de normas de gradientes a lo largo del entrenamiento.

    Si se proveen layer_norms_history, muestra un panel adicional
    con las normas de las capas mas representativas.
    """
    num_panels = 2 if layer_norms_history else 1
    fig, axes = plt.subplots(1, num_panels, figsize=(11 * num_panels // 1.5, 5))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    if num_panels == 1:
        axes = [axes]

    # Panel 1: norma global con banda de percentiles para ver tendencia
    ax_global = axes[0]
    ax_global.set_facecolor(AXES_BACKGROUND)
    global_arr  = np.array(global_norms)
    steps_arr   = np.array(steps)

    # Linea cruda muy transparente
    ax_global.semilogy(steps_arr, global_arr, color=COLOR_EMERALD, alpha=0.15, linewidth=0.7)

    # Suavizado de tendencia con ventana movil en espacio log
    window = max(10, len(global_arr) // 40)
    if len(global_arr) >= window:
        log_norms    = np.log10(np.maximum(global_arr, 1e-10))
        kernel       = np.ones(window) / window
        smooth_log   = np.convolve(log_norms, kernel, mode="valid")
        smooth_steps = steps_arr[window // 2: window // 2 + len(smooth_log)]
        ax_global.semilogy(smooth_steps, 10 ** smooth_log,
                           color=COLOR_EMERALD, linewidth=2.2, label="Norma global (tendencia)")

        # Banda P25-P75 para mostrar dispersion
        w2 = max(window, len(global_arr) // 10)
        p25 = np.array([np.percentile(global_arr[max(0,i-w2):i+w2], 25) for i in range(len(global_arr))])
        p75 = np.array([np.percentile(global_arr[max(0,i-w2):i+w2], 75) for i in range(len(global_arr))])
        ax_global.fill_between(steps_arr, p25, p75, color=COLOR_EMERALD, alpha=0.12, label="P25-P75")
    else:
        ax_global.semilogy(steps_arr, global_arr, color=COLOR_EMERALD, linewidth=2.0, label="Norma global")

    add_accent_line(ax_global, 1.0, "h", "Umbral clip=1.0", COLOR_AMBER)

    # Anotacion del valor final
    ax_global.annotate(
        f" {global_arr[-1]:.3f}",
        xy=(steps_arr[-1], global_arr[-1]),
        color=COLOR_EMERALD, fontsize=8.5, va="center",
    )

    style_ax(ax_global, xlabel="Paso", ylabel="||∇||₂  (escala log)",
             title="Norma global de gradientes", yscale="log")
    ax_global.legend()

    # Panel 2: normas por capa
    if layer_norms_history:
        ax_layers = axes[1]
        ax_layers.set_facecolor(AXES_BACKGROUND)
        colors = plt.cm.cool(np.linspace(0.2, 0.9, min(max_layers_to_show, len(layer_norms_history))))
        for (layer_name, norms), color in zip(
            list(layer_norms_history.items())[:max_layers_to_show], colors
        ):
            short_name = layer_name.split(".")[-2] + "." + layer_name.split(".")[-1]
            ax_layers.semilogy(steps, norms, color=color, linewidth=1.4, alpha=0.85, label=short_name)
        style_ax(ax_layers, xlabel="Paso", ylabel="||∇||₂  (escala log)",
                 title="Normas por capa (seleccion)", yscale="log")
        ax_layers.legend(fontsize=7, loc="upper right", ncol=2)

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.0)
    return fig


def plot_gradient_flow_heatmap(
    layer_names: List[str],
    gradient_norms_matrix: np.ndarray,
    checkpoint_steps: List[int],
    title: str = "🌡️ Mapa de Calor: Flujo de Gradientes",
) -> plt.Figure:
    """Mapa de calor (capas x pasos) de normas de gradientes.

    Permite detectar vanishing/exploding gradients visualmente.

    Parameters
    ----------
    gradient_norms_matrix : shape (num_layers, num_checkpoints)
    """
    fig, ax = plt.subplots(figsize=(13, max(4.0, len(layer_names) * 0.25)))
    fig.patch.set_facecolor(BACKGROUND_DARK)
    ax.set_facecolor(AXES_BACKGROUND)

    log_norms = np.log10(np.maximum(gradient_norms_matrix, 1e-10))

    cmap = plt.cm.RdYlGn  # rojo=bajo (vanishing), verde=normal, amarillo=alto
    im = ax.imshow(log_norms, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=np.percentile(log_norms, 5), vmax=np.percentile(log_norms, 95))

    colorbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    colorbar.set_label("log₁₀(||∇||₂)", color=TEXT_SECONDARY)
    colorbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY)
    plt.setp(colorbar.ax.yaxis.get_ticklabels(), color=TEXT_SECONDARY)

    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels([n.replace(".", "\n", 1) for n in layer_names], fontsize=7)
    ax.set_xticks(range(0, len(checkpoint_steps), max(1, len(checkpoint_steps) // 8)))
    ax.set_xticklabels(
        [str(checkpoint_steps[i]) for i in range(0, len(checkpoint_steps), max(1, len(checkpoint_steps) // 8))],
        fontsize=8,
    )
    style_ax(ax, xlabel="Paso de entrenamiento", ylabel="Capa", title=title)
    plt.tight_layout()
    return fig
