"""Recupera metricas del run actual desde TensorBoard y las guarda como JSONL.

Util cuando el entrenamiento ya arranco sin MetricsLogger.
Extrae los escalares de los archivos .tfevents y los escribe en
checkpoints/<dataset>/training_metrics.jsonl con el mismo formato.

Uso:
    python scripts/recover_metrics_from_tb.py --checkpoint_dir checkpoints/cifar10
    python scripts/recover_metrics_from_tb.py --checkpoint_dir checkpoints/mnist
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def recover_from_tensorboard(checkpoint_dir: str) -> None:
    tb_log_dir = os.path.join(checkpoint_dir, "tb_logs")
    if not os.path.exists(tb_log_dir):
        print(f"No se encontro tb_logs en {checkpoint_dir}")
        print("Verifica que el directorio del checkpoint sea correcto.")
        return

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("TensorBoard no instalado o no importable como libreria.")
        print("Instalar con: pip install tensorboard")
        return

    print(f"Leyendo TensorBoard desde: {tb_log_dir}")
    ea = EventAccumulator(tb_log_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    print(f"Tags encontrados: {tags}")

    train_loss_data   = {}
    val_loss_data     = {}
    grad_global_data  = {}

    if "train/loss" in tags:
        for event in ea.Scalars("train/loss"):
            train_loss_data[event.step] = event.value

    if "val/loss" in tags:
        for event in ea.Scalars("val/loss"):
            val_loss_data[event.step] = event.value

    if "train/gradient_norm_global" in tags:
        for event in ea.Scalars("train/gradient_norm_global"):
            grad_global_data[event.step] = event.value

    # Construir registros combinados
    output_path = os.path.join(checkpoint_dir, "training_metrics.jsonl")
    existing_steps = set()

    # No sobreescribir si ya hay datos
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        existing_steps.add((rec.get("type"), rec.get("step")))
                    except Exception:
                        pass
        print(f"JSONL existente con {len(existing_steps)} registros — solo se agregaran los nuevos.")

    records_added = 0
    with open(output_path, "a", encoding="utf-8") as f:
        # Registros de train
        all_train_steps = sorted(set(train_loss_data.keys()) | set(grad_global_data.keys()))
        for step in all_train_steps:
            if ("train", step) in existing_steps:
                continue
            record = {
                "step": step,
                "type": "train",
                "loss": round(train_loss_data.get(step, float("nan")), 6),
                "grad_global": round(grad_global_data.get(step, float("nan")), 6),
                "time_s": None,
                "_source": "recovered_from_tb",
            }
            f.write(json.dumps(record) + "\n")
            records_added += 1

        # Registros de val
        for step in sorted(val_loss_data.keys()):
            if ("val", step) in existing_steps:
                continue
            record = {
                "step": step,
                "type": "val",
                "loss": round(val_loss_data[step], 6),
                "_source": "recovered_from_tb",
            }
            f.write(json.dumps(record) + "\n")
            records_added += 1

    print(f"Registros agregados al JSONL: {records_added}")
    print(f"  train steps: {len(all_train_steps)}")
    print(f"  val steps:   {len(val_loss_data)}")
    print(f"Archivo: {output_path}")
    print()
    print("Ahora puedes generar todas las graficas con:")
    print(f"  python scripts/plot_from_logs.py --checkpoint_dir {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()
    recover_from_tensorboard(args.checkpoint_dir)
