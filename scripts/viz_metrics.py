"""Visualizaciones de metricas de evaluacion.

  - plot_fid_vs_training_steps   : curva FID a lo largo del entrenamiento
  - plot_fid_vs_sampling_steps   : curva FID vs num_pasos DDPM/DDIM
  - plot_sampling_speed_comparison: tiempo DDPM vs DDIM
  - plot_metrics_radar           : radar de FID / IS / NLL / Precision / Recall
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple

from .plot_style import (
    apply_dark_style, style_ax, add_accent_line,
    BACKGROUND_DARK, AXES_BACKGROUND, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER_COLOR, COLOR_INDIGO, COLOR_BLUE, COLOR_EMERALD,
    COLOR_AMBER, COLOR_ROSE, SAMPLER_COLORS,
)

apply_dark_style()


def plot_fid_vs_training_steps(
    checkpoint_steps: List[int],
    fid_scores_ddpm: List[float],
    fid_scores_ema: Optional[List[float]] = None,
    paper_fid: float = 3.17,
    title: str = "📈 FID vs Pasos de Entrenamiento",
) -> plt.Figure:
    """Curva de FID a lo largo del entrenamiento.

    Muestra la diferencia entre pesos raw y EMA, y una linea
    de referencia con el FID del paper original.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BACKGROUND_DARK)
    ax.set_facecolor(AXES_BACKGROUND)

    ax.plot(checkpoint_steps, fid_scores_ddpm, color=COLOR_ROSE, linewidth=2.0,
            marker="o", markersize=4, markevery=max(1, len(checkpoint_steps) // 10),
            label="FID (pesos raw)", alpha=0.8)

    if fid_scores_ema:
        ax.plot(checkpoint_steps, fid_scores_ema, color=COLOR_BLUE, linewidth=2.2,
                marker="s", markersize=4, markevery=max(1, len(checkpoint_steps) // 10),
                label="FID (pesos EMA)")

    add_accent_line(ax, paper_fid, "h", f"Paper FID {paper_fid} (TPU v3-8, 800k pasos)", COLOR_AMBER)
    style_ax(ax, xlabel="Pasos de entrenamiento", ylabel="FID (↓ mejor)",
             title=title)
    ax.legend()
    ax.invert_yaxis()

    note = "FID calculado con 50k muestras, pesos EMA, InceptionV3"
    ax.text(0.02, 0.04, note, transform=ax.transAxes,
            color=TEXT_SECONDARY, fontsize=8, alpha=0.8)

    plt.tight_layout()
    return fig


def plot_fid_vs_sampling_steps(
    ddpm_steps: List[int],
    ddpm_fid_scores: List[float],
    ddim_steps: List[int],
    ddim_fid_scores: List[float],
    ddim_times_seconds: Optional[List[float]] = None,
    title: str = "⚡ FID vs Pasos de Muestreo — DDPM vs DDIM",
) -> plt.Figure:
    """Curva FID vs num_pasos para DDPM y DDIM.

    Si se proveen tiempos, agrega un segundo eje con segundos/imagen.
    """
    fig, ax1 = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BACKGROUND_DARK)
    ax1.set_facecolor(AXES_BACKGROUND)

    ax1.plot(ddpm_steps, ddpm_fid_scores, color=COLOR_ROSE, linewidth=2.2,
             marker="o", markersize=6, label="DDPM")
    ax1.plot(ddim_steps, ddim_fid_scores, color=COLOR_BLUE, linewidth=2.2,
             marker="s", markersize=6, label="DDIM (eta=0)")

    style_ax(ax1, xlabel="Pasos de muestreo", ylabel="FID (↓ mejor)", title=title)
    ax1.legend(loc="upper right")
    ax1.invert_yaxis()

    if ddim_times_seconds:
        ax2 = ax1.twinx()
        ax2.set_facecolor("none")
        ax2.plot(ddim_steps, ddim_times_seconds, color=COLOR_EMERALD, linewidth=1.5,
                 linestyle="--", marker="^", markersize=5, label="Tiempo DDIM (s/imagen)")
        ax2.set_ylabel("Segundos por imagen", color=COLOR_EMERALD)
        ax2.tick_params(axis="y", colors=COLOR_EMERALD)
        ax2.legend(loc="lower right")

    plt.tight_layout()
    return fig


def plot_sampling_speed_comparison(
    benchmark_results: Dict[str, Dict],
    title: str = "🏎️ Comparativa de Velocidad — DDPM vs DDIM",
) -> plt.Figure:
    """Barras horizontales: tiempo medio de muestreo por configuracion.

    benchmark_results: dict {"DDPM T=1000": {...}, "DDIM S=100": {...}, ...}
    """
    labels = list(benchmark_results.keys())
    mean_times = [v["mean_seconds"] for v in benchmark_results.values()]
    std_times = [v.get("std_seconds", 0) for v in benchmark_results.values()]
    colors = [COLOR_ROSE if "DDPM" in lbl else COLOR_BLUE for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, max(4.0, len(labels) * 0.7)))
    fig.patch.set_facecolor(BACKGROUND_DARK)
    ax.set_facecolor(AXES_BACKGROUND)

    y_positions = np.arange(len(labels))
    bars = ax.barh(y_positions, mean_times, xerr=std_times, color=colors,
                   alpha=0.85, edgecolor=BORDER_COLOR, height=0.6, capsize=4)

    for bar, mean_t, std_t in zip(bars, mean_times, std_times):
        ax.text(mean_t + max(std_times) * 0.05, bar.get_y() + bar.get_height() / 2,
                f"  {mean_t:.1f}s", va="center", color=TEXT_SECONDARY, fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    style_ax(ax, xlabel="Tiempo de muestreo (s, batch=16)", title=title)
    plt.tight_layout()
    return fig


def plot_metrics_summary_table(
    metrics_dict: Dict[str, float],
    paper_values: Optional[Dict[str, float]] = None,
    title: str = "📋 Resumen de Métricas de Evaluación",
) -> plt.Figure:
    """Tabla visual con nuestras metricas vs. paper.

    metrics_dict y paper_values: {"FID": val, "IS": val, "NLL": val, ...}
    """
    metric_names = list(metrics_dict.keys())
    our_values = [metrics_dict[k] for k in metric_names]
    paper_vals = [paper_values.get(k, float("nan")) if paper_values else float("nan") for k in metric_names]

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(metric_names) * 0.6 + 1.5)))
    fig.patch.set_facecolor(BACKGROUND_DARK)
    ax.set_facecolor(AXES_BACKGROUND)

    x = np.arange(len(metric_names))
    width = 0.35
    ax.bar(x - width / 2, our_values, width, label="Nuestra implementacion",
           color=COLOR_INDIGO, alpha=0.85, edgecolor=BORDER_COLOR)

    if paper_values:
        ax.bar(x + width / 2, paper_vals, width, label="Paper (Ho et al.)",
               color=COLOR_AMBER, alpha=0.85, edgecolor=BORDER_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15, ha="right")
    style_ax(ax, ylabel="Valor", title=title)
    ax.legend()
    plt.tight_layout()
    return fig
