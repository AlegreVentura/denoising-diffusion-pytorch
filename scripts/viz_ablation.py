"""Visualizaciones para ablaciones del proyecto.

  - plot_schedule_forward_comparison : mosaico imagenes ruidosas por schedule
  - plot_schedule_snr_comparison     : curvas SNR para los 3 schedules
  - plot_prediction_type_comparison  : epsilon vs x0 vs v (curvas de loss)
  - plot_ema_ablation                : con/sin EMA en FID y calidad visual
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from .plot_style import (
    apply_dark_style, style_ax, add_accent_line,
    BACKGROUND_DARK, AXES_BACKGROUND, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER_COLOR, COLOR_INDIGO, COLOR_BLUE, COLOR_EMERALD,
    COLOR_AMBER, COLOR_ROSE, COLOR_VIOLET,
    SCHEDULE_COLORS, PREDICTION_COLORS, tensor_to_uint8_image,
)

apply_dark_style()


def plot_schedule_forward_comparison(
    forward_samples: Dict[str, Dict[int, "torch.Tensor"]],
    timesteps_to_show: List[int],
    title: str = "🔀 Ablación: Proceso Forward por Schedule de Ruido",
) -> plt.Figure:
    """Mosaico: cada fila es un schedule, cada columna es un timestep.

    Parameters
    ----------
    forward_samples : {schedule_name: {t: tensor (C,H,W)}}
    """
    schedule_names = list(forward_samples.keys())
    nrows = len(schedule_names)
    ncols = 1 + len(timesteps_to_show)

    schedule_label_map = {
        "linear":  "Lineal",
        "cosine":  "Coseno",
        "sigmoid": "Sigmoide",
    }

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.2))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, schedule_name in enumerate(schedule_names):
        for col_idx, t in enumerate([-1] + timesteps_to_show):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(BACKGROUND_DARK)
            ax.axis("off")

            if col_idx == 0:
                label = schedule_label_map.get(schedule_name, schedule_name)
                ax.text(0.5, 0.5, label, transform=ax.transAxes,
                        ha="center", va="center",
                        color=SCHEDULE_COLORS.get(schedule_name, COLOR_INDIGO),
                        fontsize=11, fontweight="bold")
            else:
                t_val = timesteps_to_show[col_idx - 1]
                img_tensor = forward_samples[schedule_name].get(t_val)
                if img_tensor is not None:
                    if img_tensor.shape[0] == 1:
                        img_array = tensor_to_uint8_image(img_tensor.repeat(3, 1, 1))
                    else:
                        img_array = tensor_to_uint8_image(img_tensor)
                    ax.imshow(img_array)
                    if row_idx == 0:
                        ax.set_title(f"t={t_val}", color=TEXT_SECONDARY, fontsize=9)

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.2)
    return fig


def plot_schedule_snr_comparison(
    snr_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    half_signal_timesteps: Dict[str, int],
    title: str = "📊 SNR por Schedule — Velocidad de Destrucción de Información",
) -> plt.Figure:
    """Compara las curvas SNR(t) de los 3 schedules con marcadores del t_{SNR=1}."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    for ax in axes:
        ax.set_facecolor(AXES_BACKGROUND)

    ax_lin, ax_log = axes

    for schedule_name, (timesteps, snr) in snr_data.items():
        color = SCHEDULE_COLORS.get(schedule_name, COLOR_INDIGO)
        label = {"linear": "Lineal", "cosine": "Coseno", "sigmoid": "Sigmoide"}.get(schedule_name, schedule_name)
        ax_lin.plot(timesteps, snr, color=color, linewidth=2.2, label=label)
        ax_log.semilogy(timesteps, snr + 1e-8, color=color, linewidth=2.2, label=label)

        t_half = half_signal_timesteps.get(schedule_name)
        if t_half is not None:
            snr_at_half = snr[t_half] if t_half < len(snr) else 1.0
            ax_lin.axvline(t_half, color=color, linestyle=":", linewidth=1.2, alpha=0.6)
            ax_log.axvline(t_half, color=color, linestyle=":", linewidth=1.2, alpha=0.6)

    add_accent_line(ax_lin, 1.0, "h", "SNR = 1 (señal = ruido)", COLOR_AMBER)
    add_accent_line(ax_log, 1.0, "h", "SNR = 1", COLOR_AMBER)

    style_ax(ax_lin, xlabel="Timestep t", ylabel="SNR(t)", title="Escala lineal")
    style_ax(ax_log, xlabel="Timestep t", ylabel="SNR(t)  (log)", title="Escala logarítmica", yscale="log")
    ax_lin.legend()
    ax_log.legend()

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.0)
    return fig


def plot_prediction_type_loss_comparison(
    loss_curves: Dict[str, Dict],
    title: str = "🔬 Ablación: ε-prediction vs x₀-prediction vs v-prediction",
) -> plt.Figure:
    """Compara las curvas de perdida para las tres parametrizaciones.

    loss_curves: {"epsilon": {"steps": [...], "train_loss": [...], "val_loss": [...]}, ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    for ax in axes:
        ax.set_facecolor(AXES_BACKGROUND)

    ax_train, ax_val = axes
    labels_map = {
        "epsilon": "ε-prediction (DDPM)",
        "x0":      "x₀-prediction",
        "v":       "v-prediction",
    }

    for pred_type, data in loss_curves.items():
        color = PREDICTION_COLORS.get(pred_type, COLOR_INDIGO)
        label = labels_map.get(pred_type, pred_type)
        ax_train.plot(data["steps"], data["train_loss"], color=color, linewidth=2.0, label=label)
        if "val_loss" in data:
            ax_val.plot(data["steps"], data["val_loss"], color=color, linewidth=2.0, label=label)

    style_ax(ax_train, xlabel="Paso", ylabel="L_simple (MSE)", title="Loss de entrenamiento")
    style_ax(ax_val, xlabel="Paso", ylabel="L_simple (MSE)", title="Loss de validación")
    ax_train.legend()
    ax_val.legend()

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.0)
    return fig


def plot_ema_ablation(
    steps: List[int],
    fid_with_ema: List[float],
    fid_without_ema: List[float],
    title: str = "🔧 Ablación EMA — Impacto en Calidad de Generación",
) -> plt.Figure:
    """Curva FID con y sin EMA para demostrar su impacto."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BACKGROUND_DARK)
    ax.set_facecolor(AXES_BACKGROUND)

    ax.plot(steps, fid_with_ema, color=COLOR_BLUE, linewidth=2.2, label="Con EMA (decay=0.9999)")
    ax.plot(steps, fid_without_ema, color=COLOR_ROSE, linewidth=2.0, linestyle="--", label="Sin EMA (pesos raw)")

    note_text = "FID sin EMA = ~12-13; con EMA = ~3.1 (checkpoint oficial)"
    ax.text(0.02, 0.96, note_text, transform=ax.transAxes,
            color=TEXT_SECONDARY, fontsize=8.5, va="top")

    style_ax(ax, xlabel="Pasos de entrenamiento", ylabel="FID (↓ mejor)", title=title)
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    return fig
