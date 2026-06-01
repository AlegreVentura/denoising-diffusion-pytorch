import torch
import os
from pathlib import Path
from typing import Any, Dict


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.cuda.amp.GradScaler",
    metrics: Dict[str, float],
    is_best: bool = False,
) -> None:
    """Guarda modelo + EMA + optimizer + scaler + step + metricas.

    Siempre guarda 'latest.pt'; si is_best=True tambien guarda 'best.pt'.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(state, latest_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(state, best_path)

    stepped_path = os.path.join(checkpoint_dir, f"step_{step:07d}.pt")
    torch.save(state, stepped_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scaler: "torch.cuda.amp.GradScaler" = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Carga un checkpoint y restaura todos los estados."""
    state = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state["model_state_dict"])
    ema_model.load_state_dict(state["ema_state_dict"])

    if optimizer is not None and state.get("optimizer_state_dict"):
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scaler is not None and state.get("scaler_state_dict"):
        scaler.load_state_dict(state["scaler_state_dict"])

    return state
