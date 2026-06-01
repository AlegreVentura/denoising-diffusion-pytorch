import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic_cudnn: bool = False) -> None:
    """Fija semillas en todos los RNGs para reproducibilidad.

    deterministic_cudnn=True elimina no-determinismo de cuDNN
    pero reduce el rendimiento; documenta si lo activas.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
