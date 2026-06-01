"""Logger de metricas a archivo JSONL independiente de TensorBoard.

Guarda una linea JSON por evento, lo que permite:
  - Recuperar todas las metricas aunque TensorBoard no arranque
  - Importar directamente en pandas / matplotlib sin dependencias extra
  - Reanudar correctamente si el entrenamiento se interrumpe

Formato de cada linea:
  {"step": 1000, "type": "train", "loss": 0.042, "grad_global": 0.31, "time_s": 1712345678.1}
  {"step": 5000, "type": "val",   "loss": 0.047}
  {"step": 5000, "type": "grad_layers", "norms": {"input_conv.weight": 0.12, ...}}
"""
import json
import time
import os
from pathlib import Path
from typing import Dict, Optional, List


class MetricsLogger:
    """Escribe metricas a un archivo JSONL de forma incremental.

    Uso tipico:
        logger = MetricsLogger(checkpoint_dir)
        logger.log_train(step, loss, grad_norm)
        logger.log_val(step, val_loss)
        logger.log_grad_layers(step, per_layer_norms)  # en checkpoints

    Para leer los logs guardados:
        from utils.metrics_logger import load_metrics
        train_df, val_df, grad_df = load_metrics("checkpoints/cifar10")
    """

    def __init__(self, checkpoint_dir: str):
        self.log_path = os.path.join(checkpoint_dir, "training_metrics.jsonl")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a", encoding="utf-8", buffering=1)

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_train(self, step: int, loss: float, grad_global_norm: float) -> None:
        self._write({
            "step": step,
            "type": "train",
            "loss": round(loss, 6),
            "grad_global": round(grad_global_norm, 6),
            "time_s": time.time(),
        })

    def log_val(self, step: int, val_loss: float) -> None:
        self._write({
            "step": step,
            "type": "val",
            "loss": round(val_loss, 6),
        })

    def log_grad_layers(self, step: int, per_layer_norms: Dict[str, float]) -> None:
        """Guarda normas por capa. Llamar solo en checkpoints (no cada paso)."""
        self._write({
            "step": step,
            "type": "grad_layers",
            "norms": {k: round(v, 6) for k, v in per_layer_norms.items()},
        })

    def log_fid(self, step: int, fid: float, sampler: str = "ddpm", num_samples: int = 50000) -> None:
        self._write({
            "step": step,
            "type": "fid",
            "fid": round(fid, 4),
            "sampler": sampler,
            "num_samples": num_samples,
        })

    def close(self) -> None:
        self._file.close()

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Funciones de lectura
# ---------------------------------------------------------------------------

def load_metrics(checkpoint_dir: str):
    """Lee el JSONL y devuelve listas separadas por tipo.

    Returns
    -------
    train_records : list de dicts con keys step, loss, grad_global, time_s
    val_records   : list de dicts con keys step, loss
    grad_records  : list de dicts con keys step, norms (dict por capa)
    fid_records   : list de dicts con keys step, fid, sampler
    """
    log_path = os.path.join(checkpoint_dir, "training_metrics.jsonl")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"No se encontro el log en {log_path}")

    train_records: List[dict] = []
    val_records:   List[dict] = []
    grad_records:  List[dict] = []
    fid_records:   List[dict] = []

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_type = record.get("type", "")
            if record_type == "train":
                train_records.append(record)
            elif record_type == "val":
                val_records.append(record)
            elif record_type == "grad_layers":
                grad_records.append(record)
            elif record_type == "fid":
                fid_records.append(record)

    return train_records, val_records, grad_records, fid_records


def summarize_metrics(checkpoint_dir: str) -> None:
    """Imprime un resumen rapido del estado del entrenamiento."""
    try:
        train_r, val_r, grad_r, fid_r = load_metrics(checkpoint_dir)
    except FileNotFoundError:
        print(f"Sin log en {checkpoint_dir}")
        return

    print(f"Directorio: {checkpoint_dir}")
    print(f"  Pasos de train loggeados : {len(train_r)}")
    if train_r:
        last = train_r[-1]
        print(f"  Ultimo paso              : {last['step']}")
        print(f"  Loss actual (train)      : {last['loss']:.6f}")
        print(f"  Norma global de grad     : {last['grad_global']:.4f}")

    if val_r:
        last_val = val_r[-1]
        print(f"  Ultimo val loss          : {last_val['loss']:.6f}  (paso {last_val['step']})")

    if grad_r:
        print(f"  Snapshots de grad/capa   : {len(grad_r)}")

    if fid_r:
        print(f"  Evaluaciones FID         : {len(fid_r)}")
        for r in fid_r:
            print(f"    paso {r['step']:>7}: FID={r['fid']:.2f} ({r['sampler']}, {r['num_samples']} muestras)")
