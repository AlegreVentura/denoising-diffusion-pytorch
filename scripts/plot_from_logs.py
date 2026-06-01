"""Genera todas las graficas del entrenamiento leyendo el JSONL guardado.

Sin depender de TensorBoard. Guardar las figuras en checkpoints/<dataset>/plots/

Uso:
    python scripts/plot_from_logs.py --checkpoint_dir checkpoints/cifar10
    python scripts/plot_from_logs.py --checkpoint_dir checkpoints/mnist
    python scripts/plot_from_logs.py --checkpoint_dir checkpoints/cifar10 --show
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics_logger import load_metrics, summarize_metrics
from scripts.plot_style import apply_dark_style
from scripts import viz_training, viz_metrics

apply_dark_style()


def extract_arrays(train_records, val_records, grad_records):
    train_steps  = [r["step"]        for r in train_records]
    train_losses = [r["loss"]        for r in train_records]
    grad_globals = [r["grad_global"] for r in train_records]

    val_steps  = [r["step"] for r in val_records]
    val_losses = [r["loss"] for r in val_records]

    # Normas por capa: construir historial {nombre_capa: [norma_en_cada_snapshot]}
    layer_norm_history = {}
    grad_steps = []
    for record in grad_records:
        grad_steps.append(record["step"])
        for layer_name, norm_val in record["norms"].items():
            if layer_name not in layer_norm_history:
                layer_norm_history[layer_name] = []
            layer_norm_history[layer_name].append(norm_val)

    return (
        train_steps, train_losses, grad_globals,
        val_steps, val_losses,
        grad_steps, layer_norm_history,
    )


def plot_and_save(fig: plt.Figure, output_dir: str, filename: str, show: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Guardada: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(checkpoint_dir: str, show: bool = False) -> None:
    print(f"\nLeyendo metricas de: {checkpoint_dir}")
    summarize_metrics(checkpoint_dir)
    print()

    train_records, val_records, grad_records, fid_records = load_metrics(checkpoint_dir)

    if not train_records:
        print("Sin datos de entrenamiento en el log. Verifica que train.py este usando MetricsLogger.")
        return

    (
        train_steps, train_losses, grad_globals,
        val_steps, val_losses,
        grad_steps, layer_norm_history,
    ) = extract_arrays(train_records, val_records, grad_records)

    output_dir = os.path.join(checkpoint_dir, "plots")
    print(f"Guardando graficas en: {output_dir}\n")

    # ------------------------------------------------------------------
    # 1. Curvas de perdida train / val
    # ------------------------------------------------------------------
    print("Generando: loss_curves.png")
    fig = viz_training.plot_training_loss_curves(
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps if val_losses else None,
        val_losses=val_losses if val_losses else None,
        smoothing_window=max(10, len(train_steps) // 100),
        title="Curvas de Perdida — Entrenamiento vs Validacion",
    )
    plot_and_save(fig, output_dir, "loss_curves.png", show)

    # ------------------------------------------------------------------
    # 2. Norma global de gradientes
    # ------------------------------------------------------------------
    print("Generando: gradient_norms_global.png")
    fig = viz_training.plot_gradient_norms(
        steps=train_steps,
        global_norms=grad_globals,
        layer_norms_history=layer_norm_history if layer_norm_history else None,
        title="Normas de Gradiente — Global y por Capa",
    )
    plot_and_save(fig, output_dir, "gradient_norms.png", show)

    # ------------------------------------------------------------------
    # 3. Mapa de calor de gradientes por capa (si hay suficientes snapshots)
    # ------------------------------------------------------------------
    if layer_norm_history and len(grad_steps) >= 3:
        print("Generando: gradient_heatmap.png")
        layer_names_sorted = sorted(layer_norm_history.keys())
        norm_matrix = np.array([
            layer_norm_history[name] for name in layer_names_sorted
        ])
        # Recortar a las capas con mas varianza (las mas interesantes)
        layer_variance = norm_matrix.std(axis=1)
        top_indices = np.argsort(layer_variance)[-min(40, len(layer_names_sorted)):]
        top_names   = [layer_names_sorted[i] for i in top_indices]
        top_matrix  = norm_matrix[top_indices]

        fig = viz_training.plot_gradient_flow_heatmap(
            layer_names=top_names,
            gradient_norms_matrix=top_matrix,
            checkpoint_steps=grad_steps,
            title="Mapa de Calor: Flujo de Gradientes (top capas por varianza)",
        )
        plot_and_save(fig, output_dir, "gradient_heatmap.png", show)

    # ------------------------------------------------------------------
    # 4. FID vs pasos de entrenamiento (si se corrió eval.py)
    # ------------------------------------------------------------------
    if fid_records:
        print("Generando: fid_vs_steps.png")
        fid_steps_raw = [r["step"] for r in fid_records if r.get("sampler", "ddpm") == "ddpm"]
        fid_vals_raw  = [r["fid"]  for r in fid_records if r.get("sampler", "ddpm") == "ddpm"]
        fid_steps_ema = [r["step"] for r in fid_records if "ema" in r.get("sampler", "")]
        fid_vals_ema  = [r["fid"]  for r in fid_records if "ema" in r.get("sampler", "")]

        fig = viz_metrics.plot_fid_vs_training_steps(
            checkpoint_steps=fid_steps_raw or fid_steps_ema,
            fid_scores_ddpm=fid_vals_raw or fid_vals_ema,
            fid_scores_ema=fid_vals_ema if (fid_steps_raw and fid_steps_ema) else None,
        )
        plot_and_save(fig, output_dir, "fid_vs_steps.png", show)
    else:
        print("Sin datos FID todavia. Ejecutar eval.py para anadirlos al log.")
        print("  Comando: python eval.py --config configs/cifar10.yaml --checkpoint checkpoints/cifar10/best.pt")

    # ------------------------------------------------------------------
    # 5. Resumen rapido en consola
    # ------------------------------------------------------------------
    last_train = train_records[-1]
    print(f"\nResumen final:")
    print(f"  Ultimo paso        : {last_train['step']:,}")
    print(f"  Loss (train)       : {last_train['loss']:.6f}")
    print(f"  Norma grad global  : {last_train['grad_global']:.4f}")
    if val_records:
        last_val = val_records[-1]
        print(f"  Loss (val)         : {last_val['loss']:.6f}  (paso {last_val['step']:,})")
    print(f"  Graficas guardadas : {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar graficas desde el log de entrenamiento")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directorio del checkpoint (p.ej. checkpoints/cifar10)")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar graficas interactivas ademas de guardarlas")
    args = parser.parse_args()
    main(args.checkpoint_dir, args.show)
