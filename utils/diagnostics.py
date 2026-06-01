"""Diagnostico de gradientes por capa.

Llamar despues de loss.backward() y antes de optimizer.step().
Con AMP: invocar primero scaler.unscale_(optimizer) para que
los valores no esten escalados artificialmente.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


def compute_gradient_norms(model: nn.Module) -> Tuple[Dict[str, float], float]:
    """Norma L2 por capa y norma global.

    Returns
    -------
    per_layer_norms : dict {nombre_parametro: norma_L2}
    global_norm     : norma L2 sobre todos los parametros con gradiente
    """
    per_layer_norms: Dict[str, float] = {}
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            per_layer_norms[param_name] = param.grad.detach().norm(2).item()

    global_norm = sum(v ** 2 for v in per_layer_norms.values()) ** 0.5
    return per_layer_norms, global_norm


def clip_gradients_and_log(
    model: nn.Module,
    max_grad_norm: float = 1.0,
) -> Tuple[Dict[str, float], float]:
    """Recorta gradientes (in-place) y devuelve normas antes del recorte."""
    per_layer_norms, global_norm_before_clip = compute_gradient_norms(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    return per_layer_norms, global_norm_before_clip
