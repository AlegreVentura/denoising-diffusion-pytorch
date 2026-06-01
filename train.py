"""Script de entrenamiento principal para DDPM.

Uso:
    python train.py --config configs/cifar10.yaml
    python train.py --config configs/mnist.yaml --resume checkpoints/mnist/latest.pt

Caracteristicas:
  - AMP (autocast + GradScaler): ~1.5-2x speedup en RTX 3060 Ti
  - EMA con decay=0.9999
  - Checkpointing completo (modelo + EMA + optimizer + step + RNG state)
  - Diagnostico de gradientes por capa
  - Logging por TensorBoard
  - Validacion concurrente cada N pasos
"""
import argparse
import os
import sys
import math
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ddpm import GaussianDiffusion, UNet, ExponentialMovingAverage
from ddpm.diffusion import make_linear_beta_schedule, make_cosine_beta_schedule, make_sigmoid_beta_schedule
from data import get_mnist_loaders, get_fashion_mnist_loaders, get_cifar10_loaders
from utils import seed_everything
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.diagnostics import clip_gradients_and_log
from utils.metrics_logger import MetricsLogger


SCHEDULE_FACTORY = {
    "linear":  make_linear_beta_schedule,
    "cosine":  make_cosine_beta_schedule,
    "sigmoid": make_sigmoid_beta_schedule,
}

DATASET_FACTORY = {
    "mnist":         get_mnist_loaders,
    "fashion_mnist": get_fashion_mnist_loaders,
    "cifar10":       get_cifar10_loaders,
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> UNet:
    model_cfg = cfg["model"]
    return UNet(
        image_channels=cfg["dataset"]["image_channels"],
        base_channels=model_cfg["base_channels"],
        channel_multipliers=tuple(model_cfg["channel_multipliers"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        attention_resolutions=tuple(model_cfg["attention_resolutions"]),
        dropout=model_cfg["dropout"],
        num_groups=model_cfg["num_groups"],
    ).to(device)


def build_diffusion(cfg: dict) -> GaussianDiffusion:
    diff_cfg = cfg["diffusion"]
    schedule_fn = SCHEDULE_FACTORY[diff_cfg["schedule"]]
    if diff_cfg["schedule"] == "linear":
        betas = schedule_fn(diff_cfg["num_timesteps"], diff_cfg["beta_start"], diff_cfg["beta_end"])
    else:
        betas = schedule_fn(diff_cfg["num_timesteps"])
    return GaussianDiffusion(betas)


def save_sample_grid(
    model: UNet,
    diffusion: GaussianDiffusion,
    ema: ExponentialMovingAverage,
    cfg: dict,
    step: int,
    writer: SummaryWriter,
    device: torch.device,
) -> None:
    """Genera una cuadricula de muestras EMA y la sube a TensorBoard."""
    num_samples = cfg["sampling"]["num_display_samples"]
    image_size = cfg["dataset"]["image_size"]
    image_channels = cfg["dataset"]["image_channels"]

    # Usar pesos EMA para la generacion
    ema.copy_to(model.parameters())
    model.eval()

    with torch.no_grad():
        samples = diffusion.sample(
            model, batch_size=num_samples,
            image_channels=image_channels, image_size=image_size, device=str(device),
        )

    # Restaurar pesos originales
    model.train()

    # Normalizar a [0, 1] para TensorBoard
    samples_display = (samples * 0.5 + 0.5).clamp(0, 1)
    grid_nrow = int(math.sqrt(num_samples))
    from torchvision.utils import make_grid
    grid = make_grid(samples_display[:grid_nrow ** 2], nrow=grid_nrow, padding=2)
    writer.add_image("samples/ema", grid, global_step=step)


def validate(
    model: UNet,
    diffusion: GaussianDiffusion,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Calcula la loss de validacion (sin gradientes, sin AMP)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, (x_batch, _) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            x_batch = x_batch.to(device)
            loss = diffusion.compute_loss_simple(model, x_batch)
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / max(num_batches, 1)


def train(config_path: str, resume_checkpoint: str = None) -> None:
    cfg = load_config(config_path)
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")
    if device.type == "cuda":
        print(f"📟  GPU: {torch.cuda.get_device_name(0)}")

    # Construir componentes
    model = build_model(cfg, device)
    diffusion = build_diffusion(cfg)
    ema = ExponentialMovingAverage.from_model(model, decay=cfg["ema"]["decay"])

    num_params = model.count_parameters()
    print(f"🏗️  Parametros del modelo: {num_params:,}  (paper: 35.7M para CIFAR-10)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scaler = GradScaler("cuda", enabled=cfg["training"]["use_amp"])

    # DataLoaders
    dataset_name = cfg["dataset"]["name"]
    loader_fn = DATASET_FACTORY[dataset_name]
    train_loader, val_loader, _ = loader_fn(
        data_root=cfg["dataset"]["data_root"],
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
    )

    # TensorBoard + MetricsLogger independiente
    checkpoint_dir = cfg["training"]["checkpoint_dir"]
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "tb_logs"))
    metrics_logger = MetricsLogger(checkpoint_dir)

    # Reanudar si se pasa checkpoint
    start_step = 0
    if resume_checkpoint:
        state = load_checkpoint(resume_checkpoint, model, ema, optimizer, scaler, device=str(device))
        start_step = state["step"]
        print(f"▶️  Reanudando desde paso {start_step}")

    # Bucle de entrenamiento
    model.train()
    train_iter = iter(train_loader)
    total_steps = cfg["training"]["total_steps"]
    best_val_loss = float("inf")

    log_interval    = cfg["training"]["log_every_n_steps"]
    save_interval   = cfg["training"]["save_every_n_steps"]
    sample_interval = cfg["training"]["sample_every_n_steps"]
    val_interval    = cfg["training"]["val_every_n_steps"]
    max_grad_norm   = cfg["training"]["grad_clip_max_norm"]

    progress_bar = tqdm(range(start_step, total_steps), desc="Training", dynamic_ncols=True)

    for step in progress_bar:
        try:
            x_batch, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_batch, _ = next(train_iter)

        x_batch = x_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=cfg["training"]["use_amp"]):
            loss = diffusion.compute_loss_simple(model, x_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Diagnostico de gradientes (antes del recorte)
        per_layer_norms, global_norm = clip_gradients_and_log(model, max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        ema.update(model.parameters())

        # Logging: TensorBoard + JSONL
        if step % log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/gradient_norm_global", global_norm, step)
            metrics_logger.log_train(step, loss.item(), global_norm)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{global_norm:.3f}")

        # Validacion
        if step % val_interval == 0 and step > 0:
            val_loss = validate(model, diffusion, val_loader, device)
            writer.add_scalar("val/loss", val_loss, step)
            metrics_logger.log_val(step, val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # Normas por capa solo en cada checkpoint (no cada paso: demasiado volumen)
            metrics_logger.log_grad_layers(step, per_layer_norms)

            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                model=model,
                ema_model=ema,
                optimizer=optimizer,
                scaler=scaler,
                metrics={"val_loss": val_loss, "best_val_loss": best_val_loss},
                is_best=is_best,
            )

        # Muestras visuales
        if step % sample_interval == 0 and step > 0:
            save_sample_grid(model, diffusion, ema, cfg, step, writer, device)

    writer.close()
    metrics_logger.close()
    print(f"Entrenamiento completado. Checkpoints en: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar DDPM desde cero")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo YAML de configuracion")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint desde donde reanudar")
    args = parser.parse_args()
    train(args.config, args.resume)
