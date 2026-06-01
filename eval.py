"""Evaluacion del modelo entrenado.

Protocolo correcto (documentar en el informe):
  - 50,000 muestras para FID (no 10k: el FID se infla artificialmente)
  - Pesos EMA, no raw (sin EMA: FID ~12-13 en lugar de ~3.1)
  - model.eval() + torch.no_grad() (exigido por la rubrica)

Uso:
    python eval.py --config configs/cifar10.yaml --checkpoint checkpoints/cifar10/best.pt
    python eval.py --config configs/cifar10.yaml --checkpoint checkpoints/cifar10/best.pt --ddim --ddim_steps 100
"""
import argparse
import os
import math
import json
from pathlib import Path

import yaml
import torch
import torchvision

from ddpm import GaussianDiffusion, UNet, ExponentialMovingAverage, DDIMSampler
from ddpm.diffusion import make_linear_beta_schedule, make_cosine_beta_schedule, make_sigmoid_beta_schedule
from utils.checkpointing import load_checkpoint
from utils import seed_everything


SCHEDULE_FACTORY = {
    "linear":  make_linear_beta_schedule,
    "cosine":  make_cosine_beta_schedule,
    "sigmoid": make_sigmoid_beta_schedule,
}


def build_model_and_diffusion(cfg: dict, device: torch.device):
    model_cfg = cfg["model"]
    diff_cfg  = cfg["diffusion"]

    model = UNet(
        image_channels=cfg["dataset"]["image_channels"],
        base_channels=model_cfg["base_channels"],
        channel_multipliers=tuple(model_cfg["channel_multipliers"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        attention_resolutions=tuple(model_cfg["attention_resolutions"]),
        dropout=model_cfg["dropout"],
        num_groups=model_cfg["num_groups"],
    ).to(device)

    schedule_fn = SCHEDULE_FACTORY[diff_cfg["schedule"]]
    if diff_cfg["schedule"] == "linear":
        betas = schedule_fn(diff_cfg["num_timesteps"], diff_cfg["beta_start"], diff_cfg["beta_end"])
    else:
        betas = schedule_fn(diff_cfg["num_timesteps"])

    diffusion = GaussianDiffusion(betas)
    ema = ExponentialMovingAverage.from_model(model)
    return model, diffusion, ema


def compute_fid_score(
    model: UNet,
    diffusion: GaussianDiffusion,
    cfg: dict,
    num_samples: int,
    device: torch.device,
    use_ddim: bool = False,
    ddim_steps: int = 100,
    batch_size: int = 64,
) -> float:
    """Calcula FID con num_samples muestras del modelo EMA.

    Requiere torchmetrics[image] instalado.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("⚠️  torchmetrics no instalado. pip install torchmetrics[image]")
        return float("nan")

    from data import get_cifar10_loaders
    image_size = cfg["dataset"]["image_size"]
    image_channels = cfg["dataset"]["image_channels"]

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Cargar imagenes reales
    _, _, test_loader = get_cifar10_loaders(
        data_root=cfg["dataset"]["data_root"],
        batch_size=batch_size,
        num_workers=4,
    )
    print(f"📦 Cargando imagenes reales para FID...")
    real_images_loaded = 0
    model.eval()
    with torch.no_grad():
        for real_batch, _ in test_loader:
            real_batch = real_batch.to(device)
            real_batch_01 = (real_batch * 0.5 + 0.5).clamp(0, 1)
            fid_metric.update(real_batch_01, real=True)
            real_images_loaded += real_batch.shape[0]
            if real_images_loaded >= num_samples:
                break

    # Generar imagenes con el modelo
    print(f"🎨 Generando {num_samples} imagenes {'DDIM' if use_ddim else 'DDPM'}...")
    generated_count = 0
    ddim_sampler = DDIMSampler(diffusion, num_steps=ddim_steps) if use_ddim else None

    with torch.no_grad():
        while generated_count < num_samples:
            current_batch_size = min(batch_size, num_samples - generated_count)
            if use_ddim:
                samples = ddim_sampler.sample(
                    model, current_batch_size, image_channels, image_size, str(device)
                )
            else:
                samples = diffusion.sample(
                    model, current_batch_size, image_channels, image_size, str(device)
                )
            samples_01 = (samples * 0.5 + 0.5).clamp(0, 1)
            fid_metric.update(samples_01, real=False)
            generated_count += current_batch_size

    fid_score = fid_metric.compute().item()
    return fid_score


def compute_inception_score(
    model: UNet,
    diffusion: GaussianDiffusion,
    cfg: dict,
    num_samples: int,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """Calcula Inception Score (IS). Retorna (IS_mean, IS_std)."""
    try:
        from torchmetrics.image.inception import InceptionScore
    except ImportError:
        print("⚠️  torchmetrics no instalado.")
        return float("nan"), float("nan")

    image_size = cfg["dataset"]["image_size"]
    image_channels = cfg["dataset"]["image_channels"]

    is_metric = InceptionScore(normalize=True).to(device)
    model.eval()

    generated_count = 0
    with torch.no_grad():
        while generated_count < num_samples:
            current_batch_size = min(batch_size, num_samples - generated_count)
            samples = diffusion.sample(
                model, current_batch_size, image_channels, image_size, str(device)
            )
            samples_01 = (samples * 0.5 + 0.5).clamp(0, 1)
            if image_channels == 1:
                samples_01 = samples_01.repeat(1, 3, 1, 1)
            is_metric.update(samples_01)
            generated_count += current_batch_size

    is_mean, is_std = is_metric.compute()
    return is_mean.item(), is_std.item()


def evaluate(
    config_path: str,
    checkpoint_path: str,
    use_ddim: bool = False,
    ddim_steps: int = 100,
) -> None:
    cfg = load_config_yaml(config_path)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")

    model, diffusion, ema = build_model_and_diffusion(cfg, device)

    # Cargar pesos EMA del checkpoint
    state = load_checkpoint(checkpoint_path, model, ema, device=str(device))
    print(f"✅ Checkpoint cargado (paso {state['step']})")

    # Copiar pesos EMA al modelo para evaluacion
    ema.copy_to(model.parameters())
    model.eval()

    num_fid_samples = cfg["sampling"]["num_samples_for_fid"]

    print(f"\n{'='*50}")
    print(f"EVALUACION — {cfg['dataset']['name'].upper()}")
    print(f"Muestrador: {'DDIM S=' + str(ddim_steps) if use_ddim else 'DDPM T=1000'}")
    print(f"Num muestras FID: {num_fid_samples}")
    print(f"{'='*50}\n")

    results = {}

    if cfg["dataset"]["name"] == "cifar10":
        print("📊 Calculando FID...")
        fid = compute_fid_score(model, diffusion, cfg, num_fid_samples, device, use_ddim, ddim_steps)
        print(f"  FID:  {fid:.2f}  (paper: 3.17 con 800k pasos en TPU v3-8)")
        results["FID"] = fid

        print("📊 Calculando Inception Score...")
        is_mean, is_std = compute_inception_score(model, diffusion, cfg, min(10000, num_fid_samples), device)
        print(f"  IS:   {is_mean:.2f} ± {is_std:.2f}  (paper: 9.46)")
        results["IS_mean"] = is_mean
        results["IS_std"] = is_std

    # Guardar resultados
    checkpoint_dir = os.path.dirname(checkpoint_path)
    sampler_tag = f"ddim_{ddim_steps}" if use_ddim else "ddpm_1000"
    results_path = os.path.join(checkpoint_dir, f"eval_{sampler_tag}_step{state['step']}.json")
    with open(results_path, "w") as f:
        json.dump({**results, "step": state["step"], "sampler": sampler_tag}, f, indent=2)
    print(f"\n💾 Resultados guardados en: {results_path}")


def load_config_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar modelo DDPM entrenado")
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--ddim",        action="store_true", help="Usar muestreador DDIM")
    parser.add_argument("--ddim_steps",  type=int, default=100)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.ddim, args.ddim_steps)
