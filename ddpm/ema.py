"""Exponential Moving Average de pesos del modelo.

El paper usa decay=0.9999; evaluar SIEMPRE desde EMA, no desde el modelo raw.
El FID sube ~12-13 sin EMA y baja a ~3.1 con EMA (protocolo correcto).
"""
import torch
import torch.nn as nn
import copy
from typing import Iterable


class ExponentialMovingAverage:
    """Mantiene una copia EMA de los parametros del modelo.

    Parameters
    ----------
    parameters   : parametros del modelo a seguir
    decay        : factor de decaimiento (paper: 0.9999)
    warmup_steps : pasos donde se usa decay=min(decay, (1+step)/(10+step))
                   para evitar que los primeros pasos dominen
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = 0.9999,
        warmup_steps: int = 0,
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0

        self.shadow_parameters = [p.clone().detach() for p in parameters]

    def _effective_decay(self) -> float:
        if self.warmup_steps > 0:
            warmup_decay = (1.0 + self.num_updates) / (10.0 + self.num_updates)
            return min(self.decay, warmup_decay)
        return self.decay

    def update(self, parameters: Iterable[nn.Parameter]) -> None:
        """theta_ema <- decay * theta_ema + (1 - decay) * theta."""
        self.num_updates += 1
        effective_decay = self._effective_decay()
        with torch.no_grad():
            for shadow_param, live_param in zip(self.shadow_parameters, parameters):
                shadow_param.mul_(effective_decay).add_(live_param.data, alpha=1.0 - effective_decay)

    def copy_to(self, parameters: Iterable[nn.Parameter]) -> None:
        """Copia los pesos EMA al modelo (para evaluacion o generacion)."""
        with torch.no_grad():
            for shadow_param, live_param in zip(self.shadow_parameters, parameters):
                live_param.data.copy_(shadow_param.data)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_parameters": [p.clone() for p in self.shadow_parameters],
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
        self.shadow_parameters = [p.clone() for p in state["shadow_parameters"]]

    @classmethod
    def from_model(cls, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 0) -> "ExponentialMovingAverage":
        return cls(list(model.parameters()), decay=decay, warmup_steps=warmup_steps)
