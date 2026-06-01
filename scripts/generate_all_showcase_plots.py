"""Genera todas las graficas posibles del proyecto con los checkpoints actuales.

Uso:
    python scripts/generate_all_showcase_plots.py --show

Produce en checkpoints/mnist/plots/ y checkpoints/cifar10/plots/:
  [SIN MODELO]
    01_noise_schedules.png        comparativa lineal/coseno/sigmoide
    02_forward_process_mnist.png  imagen corrompida paso a paso
    02_forward_process_cifar.png
  [MNIST - modelo completo]
    03_samples_grid_mnist.png     cuadricula 8x8 de muestras finales
    04_sample_evolution_mnist.png calidad a distintos pasos de entrenamiento
    05_reverse_chain_mnist.png    cadena inversa: ruido -> digito
    06_ddim_vs_ddpm_mnist.png     comparativa velocidad y calidad
    07_interpolation_mnist.png    interpolacion en espacio latente
  [CIFAR-10 - checkpoint parcial]
    08_samples_cifar10.png        muestras con el checkpoint actual
    09_sample_evolution_cifar.png progresion desde paso 1k hasta actual
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time

from ddpm import GaussianDiffusion, UNet, ExponentialMovingAverage, DDIMSampler
from ddpm.diffusion import make_linear_beta_schedule
from extras.ablation_schedules import NoiseScheduleAblation
from extras.latent_interpolation import LatentSpaceInterpolator
from utils.checkpointing import load_checkpoint
from scripts.plot_style import apply_dark_style, BACKGROUND_DARK, AXES_BACKGROUND, TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR, COLOR_INDIGO, COLOR_BLUE, COLOR_AMBER, COLOR_EMERALD, COLOR_ROSE, COLOR_VIOLET, tensor_to_uint8_image
from scripts import viz_diffusion, viz_ablation, viz_metrics

apply_dark_style()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig, path, show):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  -> {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def load_model_from_config(config_path, checkpoint_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    model = UNet(
        image_channels=cfg["dataset"]["image_channels"],
        base_channels=mc["base_channels"],
        channel_multipliers=tuple(mc["channel_multipliers"]),
        num_res_blocks=mc["num_res_blocks"],
        attention_resolutions=tuple(mc["attention_resolutions"]),
        dropout=mc["dropout"],
        num_groups=mc["num_groups"],
    ).to(device)
    ema = ExponentialMovingAverage.from_model(model)
    load_checkpoint(checkpoint_path, model, ema, device=device)
    ema.copy_to(model.parameters())
    model.eval()
    betas = make_linear_beta_schedule(
        cfg["diffusion"]["num_timesteps"],
        cfg["diffusion"]["beta_start"],
        cfg["diffusion"]["beta_end"],
    )
    diffusion = GaussianDiffusion(betas)
    return model, diffusion, cfg


def make_image_grid(image_tensors, nrows, ncols, title, caption_list=None):
    num = min(len(image_tensors), nrows * ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6 + 0.5))
    fig.patch.set_facecolor(BACKGROUND_DARK)
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
    for idx in range(nrows * ncols):
        ax = axes_flat[idx]
        ax.set_facecolor(BACKGROUND_DARK)
        ax.axis("off")
        if idx < num:
            t = image_tensors[idx]
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            ax.imshow(tensor_to_uint8_image(t))
            if caption_list and idx < len(caption_list):
                ax.set_title(caption_list[idx], color=TEXT_SECONDARY, fontsize=7, pad=2)
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.15)
    return fig


# ---------------------------------------------------------------------------
# 01 - Comparativa de schedules (sin modelo)
# ---------------------------------------------------------------------------

def plot_noise_schedules(out_dir, show):
    print("\n[1/9] Comparativa de schedules de ruido...")
    ablation = NoiseScheduleAblation(1000)
    data = {}
    for name in ("linear", "cosine", "sigmoid"):
        d = ablation.diffusion_objects[name]
        ab = d.alphas_cumprod.numpy()
        data[name] = {
            "timesteps": np.arange(1000),
            "betas": d.betas.numpy(),
            "alphas_cumprod": ab,
            "snr": ab / (1.0 - ab + 1e-8),
        }
    fig = viz_diffusion.plot_noise_schedule_overview(
        data, title="Comparativa de Schedules de Ruido: beta, alpha_bar, SNR"
    )
    save(fig, os.path.join(out_dir, "01_noise_schedules.png"), show)


# ---------------------------------------------------------------------------
# 02 - Proceso forward sobre imagen real (sin modelo)
# ---------------------------------------------------------------------------

def plot_forward_process(dataset_name, image_channels, image_size, out_dir, show):
    print(f"\n[2/9] Proceso forward sobre {dataset_name}...")
    from torchvision import datasets, transforms
    norm = transforms.Normalize([0.5] * image_channels, [0.5] * image_channels)
    tf = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), norm])
    try:
        if dataset_name == "mnist":
            ds = datasets.MNIST("data/raw", train=False, download=True, transform=tf)
        else:
            ds = datasets.CIFAR10("data/raw", train=False, download=True, transform=tf)
        x_clean = ds[3][0]
    except Exception:
        x_clean = torch.randn(image_channels, image_size, image_size) * 0.3

    betas = make_linear_beta_schedule(1000, 1e-4, 0.02)
    diff = GaussianDiffusion(betas)
    timesteps_to_show = [0, 50, 100, 200, 400, 600, 800, 999]
    noisy = {}
    noise = torch.randn_like(x_clean.unsqueeze(0))
    for t in timesteps_to_show[1:]:
        tb = torch.tensor([t])
        noisy[t] = diff.q_sample(x_clean.unsqueeze(0), tb, noise)[0]

    fig = viz_diffusion.plot_forward_process_strip(
        x_clean, noisy,
        title=f"Proceso Forward — {dataset_name.upper()} (schedule lineal)",
    )
    save(fig, os.path.join(out_dir, f"02_forward_process_{dataset_name}.png"), show)


# ---------------------------------------------------------------------------
# 03 - Cuadricula de muestras finales
# ---------------------------------------------------------------------------

def plot_samples_grid(model, diffusion, cfg, out_dir, tag, show):
    print(f"\n[3/9] Cuadricula de muestras — {tag}...")
    ic = cfg["dataset"]["image_channels"]
    isz = cfg["dataset"]["image_size"]
    with torch.no_grad():
        samples = diffusion.sample(model, 64, ic, isz, DEVICE)
    imgs = [samples[i].cpu() for i in range(64)]
    fig = make_image_grid(
        imgs, 8, 8,
        f"64 muestras DDPM — {tag.upper()} (pesos EMA)",
    )
    save(fig, os.path.join(out_dir, f"03_samples_grid_{tag}.png"), show)


# ---------------------------------------------------------------------------
# 04 - Evolucion de la calidad a distintos pasos de entrenamiento
# ---------------------------------------------------------------------------

def plot_sample_evolution(config_path, checkpoint_dir, dataset_name, out_dir, show):
    print(f"\n[4/9] Evolucion de calidad por paso — {dataset_name}...")
    all_checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("step_") and f.endswith(".pt")
    ])
    if not all_checkpoints:
        print("  Sin checkpoints por paso.")
        return

    # Elegir ~6 checkpoints distribuidos uniformemente
    indices = np.linspace(0, len(all_checkpoints) - 1, min(6, len(all_checkpoints))).astype(int)
    selected = [all_checkpoints[i] for i in indices]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    ic  = cfg["dataset"]["image_channels"]
    isz = cfg["dataset"]["image_size"]
    betas = make_linear_beta_schedule(
        cfg["diffusion"]["num_timesteps"],
        cfg["diffusion"]["beta_start"],
        cfg["diffusion"]["beta_end"],
    )
    diff = GaussianDiffusion(betas)

    num_checkpoints = len(selected)
    fig, axes = plt.subplots(4, num_checkpoints, figsize=(num_checkpoints * 2.0, 9))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    # Usar la misma semilla para todos los checkpoints -> comparacion justa
    fixed_noise = torch.randn(4, ic, isz, isz, device=DEVICE)

    for col_idx, ckpt_name in enumerate(selected):
        step_num = int(ckpt_name.replace("step_", "").replace(".pt", ""))
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        model_tmp = UNet(
            image_channels=ic,
            base_channels=mc["base_channels"],
            channel_multipliers=tuple(mc["channel_multipliers"]),
            num_res_blocks=mc["num_res_blocks"],
            attention_resolutions=tuple(mc["attention_resolutions"]),
            dropout=mc["dropout"],
            num_groups=mc["num_groups"],
        ).to(DEVICE)
        ema_tmp = ExponentialMovingAverage.from_model(model_tmp)
        load_checkpoint(ckpt_path, model_tmp, ema_tmp, device=DEVICE)
        ema_tmp.copy_to(model_tmp.parameters())
        model_tmp.eval()

        with torch.no_grad():
            # DDIM 50 pasos para velocidad (misma calidad visual, 20x mas rapido)
            ddim_fast = DDIMSampler(diff, num_steps=50, eta=0.0)
            shape = (4, cfg["dataset"]["image_channels"],
                     cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])
            x = fixed_noise[:4].clone()
            for step_idx in reversed(range(len(ddim_fast.ddim_timesteps))):
                x = ddim_fast.ddim_step(model_tmp, x, step_idx)

        for row_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(BACKGROUND_DARK)
            ax.axis("off")
            img_t = x[row_idx].cpu()
            if img_t.shape[0] == 1:
                img_t = img_t.repeat(3, 1, 1)
            ax.imshow(tensor_to_uint8_image(img_t))
            if row_idx == 0:
                ax.set_title(f"paso\n{step_num:,}", color=COLOR_INDIGO, fontsize=9, fontweight="bold")

        del model_tmp, ema_tmp

    fig.suptitle(
        f"Evolucion de calidad — {dataset_name.upper()} (misma semilla, distintos checkpoints)",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=0.2)
    save(fig, os.path.join(out_dir, f"04_sample_evolution_{dataset_name}.png"), show)


# ---------------------------------------------------------------------------
# 05 - Cadena inversa: ruido -> imagen (progressive generation)
# ---------------------------------------------------------------------------

def plot_reverse_chain(model, diffusion, cfg, out_dir, tag, show):
    print(f"\n[5/9] Cadena inversa — {tag}...")
    ic  = cfg["dataset"]["image_channels"]
    isz = cfg["dataset"]["image_size"]

    num_frames = 10
    save_at = list(range(0, diffusion.num_timesteps, diffusion.num_timesteps // num_frames))

    with torch.no_grad():
        _, saved_frames = diffusion.sample_progressive(
            model, batch_size=4, image_channels=ic, image_size=isz,
            save_at_timesteps=save_at, device=DEVICE,
        )

    saved_frames_sorted = sorted(saved_frames, key=lambda x: x[0], reverse=True)

    num_cols = len(saved_frames_sorted)
    fig, axes = plt.subplots(4, num_cols, figsize=(num_cols * 1.8, 8))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    for col_idx, (t_val, batch_tensor) in enumerate(saved_frames_sorted):
        for row_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(BACKGROUND_DARK)
            ax.axis("off")
            img_t = batch_tensor[row_idx]
            if img_t.shape[0] == 1:
                img_t = img_t.repeat(3, 1, 1)
            ax.imshow(tensor_to_uint8_image(img_t))
            if row_idx == 0:
                ax.set_title(f"t={t_val}", color=TEXT_SECONDARY, fontsize=8)

    fig.suptitle(
        f"Cadena Inversa — {tag.upper()}: ruido puro -> imagen generada",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=0.15)
    save(fig, os.path.join(out_dir, f"05_reverse_chain_{tag}.png"), show)


# ---------------------------------------------------------------------------
# 06 - DDIM vs DDPM: calidad y velocidad
# ---------------------------------------------------------------------------

def plot_ddim_comparison(model, diffusion, cfg, out_dir, tag, show):
    print(f"\n[6/9] DDIM vs DDPM — {tag}...")
    ic  = cfg["dataset"]["image_channels"]
    isz = cfg["dataset"]["image_size"]

    step_configs = [1000, 100, 50, 20, 10]
    timing = {}
    sample_grids = {}

    fixed_noise = torch.randn(8, ic, isz, isz, device=DEVICE)

    for num_steps in step_configs:
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()

        with torch.no_grad():
            if num_steps == 1000:
                samples = diffusion.sample(model, 8, ic, isz, DEVICE)
                label = "DDPM T=1000"
            else:
                ddim = DDIMSampler(diffusion, num_steps=num_steps, eta=0.0)
                samples = ddim.sample(model, 8, ic, isz, DEVICE)
                label = f"DDIM S={num_steps}"

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        elapsed = time.perf_counter() - t0
        timing[label] = elapsed
        sample_grids[label] = [samples[i].cpu() for i in range(8)]

    # Panel: muestras + barras de tiempo
    num_configs = len(step_configs)
    fig = plt.figure(figsize=(num_configs * 2.2, 12))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    # Fila de imagenes por configuracion
    for col_idx, (label, imgs) in enumerate(sample_grids.items()):
        for row_idx in range(4):
            ax = fig.add_subplot(6, num_configs, row_idx * num_configs + col_idx + 1)
            ax.set_facecolor(BACKGROUND_DARK)
            ax.axis("off")
            img_t = imgs[row_idx]
            if img_t.shape[0] == 1:
                img_t = img_t.repeat(3, 1, 1)
            ax.imshow(tensor_to_uint8_image(img_t))
            if row_idx == 0:
                t_val = timing[label]
                ax.set_title(f"{label}\n{t_val:.1f}s", color=COLOR_BLUE if "DDIM" in label else COLOR_ROSE,
                             fontsize=8, fontweight="bold")

    # Panel de speedup
    ax_bar = fig.add_subplot(6, 1, 6)
    ax_bar.set_facecolor(AXES_BACKGROUND)
    labels_list = list(timing.keys())
    times_list  = list(timing.values())
    colors_list = [COLOR_ROSE if "DDPM" in l else COLOR_BLUE for l in labels_list]
    bars = ax_bar.bar(labels_list, times_list, color=colors_list, alpha=0.85, edgecolor=BORDER_COLOR)
    for bar, t_val in zip(bars, times_list):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{t_val:.1f}s", ha="center", color=TEXT_SECONDARY, fontsize=8)
    ax_bar.set_ylabel("Segundos (8 imagenes)", color=TEXT_SECONDARY, fontsize=9)
    ax_bar.tick_params(colors=TEXT_SECONDARY)
    for spine in ax_bar.spines.values():
        spine.set_color(BORDER_COLOR)
    ax_bar.set_facecolor(AXES_BACKGROUND)
    ddpm_time = timing.get("DDPM T=1000", 1)
    for label, t_val in timing.items():
        if "DDIM" in label:
            speedup = ddpm_time / t_val
            print(f"  {label}: {t_val:.2f}s  ({speedup:.1f}x speedup vs DDPM)")

    fig.suptitle(f"DDPM vs DDIM — {tag.upper()} (misma semilla, pesos EMA)",
                 color=TEXT_PRIMARY, fontsize=12, fontweight="bold")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, f"06_ddim_vs_ddpm_{tag}.png"), show)


# ---------------------------------------------------------------------------
# 07 - Interpolacion en espacio latente
# ---------------------------------------------------------------------------

def plot_interpolation(model, diffusion, cfg, out_dir, tag, show):
    print(f"\n[7/9] Interpolacion en espacio latente — {tag}...")
    from torchvision import datasets, transforms
    ic  = cfg["dataset"]["image_channels"]
    isz = cfg["dataset"]["image_size"]

    norm = transforms.Normalize([0.5] * ic, [0.5] * ic)
    tf   = transforms.Compose([transforms.Resize(isz), transforms.ToTensor(), norm])
    try:
        if "mnist" in cfg["dataset"]["name"]:
            ds = datasets.MNIST("data/raw", train=False, download=True, transform=tf)
        else:
            ds = datasets.CIFAR10("data/raw", train=False, download=True, transform=tf)
        x_a = ds[0][0].unsqueeze(0).to(DEVICE)
        x_b = ds[7][0].unsqueeze(0).to(DEVICE)
    except Exception:
        x_a = torch.randn(1, ic, isz, isz, device=DEVICE)
        x_b = torch.randn(1, ic, isz, isz, device=DEVICE)

    interpolator = LatentSpaceInterpolator(diffusion, interpolation_t=400, num_interpolation_steps=9)
    decoded, lambdas = interpolator.generate_interpolation_grid(
        model, x_a, x_b, noise_seed=42, device=DEVICE, ddim_steps=50,
    )

    num_imgs = len(decoded)
    fig, axes = plt.subplots(1, num_imgs + 2, figsize=((num_imgs + 2) * 2.0, 2.6))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    for ax, label, img_t in [
        (axes[0], "Original A", x_a[0].cpu()),
        *[(axes[i + 1], f"lambda={l:.2f}", decoded[i][0]) for i, l in enumerate(lambdas)],
        (axes[-1], "Original B", x_b[0].cpu()),
    ]:
        ax.set_facecolor(BACKGROUND_DARK)
        ax.axis("off")
        if img_t.shape[0] == 1:
            img_t = img_t.repeat(3, 1, 1)
        ax.imshow(tensor_to_uint8_image(img_t))
        ax.set_title(label, color=COLOR_VIOLET, fontsize=7.5, fontweight="bold")

    fig.suptitle(
        f"Interpolacion en Espacio Latente — {tag.upper()} (t_interp=400, DDIM S=50)",
        color=TEXT_PRIMARY, fontsize=11, fontweight="bold", y=1.04,
    )
    plt.tight_layout(pad=0.2)
    save(fig, os.path.join(out_dir, f"07_interpolation_{tag}.png"), show)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(show: bool):
    mnist_cfg      = "configs/mnist.yaml"
    cifar_cfg      = "configs/cifar10.yaml"
    mnist_ckpt     = "checkpoints/mnist/best.pt"
    cifar_ckpt     = "checkpoints/cifar10/latest.pt"
    mnist_out      = "checkpoints/mnist/plots"
    cifar_out      = "checkpoints/cifar10/plots"

    # ---- Graficas sin modelo ----
    plot_noise_schedules(cifar_out, show)
    plot_forward_process("mnist",  1, 28, mnist_out, show)
    plot_forward_process("cifar10", 3, 32, cifar_out, show)

    # ---- MNIST (modelo completo) ----
    if os.path.exists(mnist_ckpt):
        model_m, diff_m, cfg_m = load_model_from_config(mnist_cfg, mnist_ckpt, DEVICE)
        plot_samples_grid(model_m, diff_m, cfg_m, mnist_out, "mnist", show)
        plot_sample_evolution(mnist_cfg, "checkpoints/mnist", "mnist", mnist_out, show)
        plot_reverse_chain(model_m, diff_m, cfg_m, mnist_out, "mnist", show)
        plot_ddim_comparison(model_m, diff_m, cfg_m, mnist_out, "mnist", show)
        plot_interpolation(model_m, diff_m, cfg_m, mnist_out, "mnist", show)
        del model_m
    else:
        print("Sin checkpoint de MNIST, saltando graficas del modelo.")

    # ---- CIFAR-10 (checkpoint parcial) ----
    if os.path.exists(cifar_ckpt):
        model_c, diff_c, cfg_c = load_model_from_config(cifar_cfg, cifar_ckpt, DEVICE)
        plot_samples_grid(model_c, diff_c, cfg_c, cifar_out, "cifar10", show)
        plot_sample_evolution(cifar_cfg, "checkpoints/cifar10", "cifar10", cifar_out, show)
        plot_reverse_chain(model_c, diff_c, cfg_c, cifar_out, "cifar10", show)
        del model_c
    else:
        print("Sin checkpoint de CIFAR-10.")

    print("\nListo. Graficas guardadas en:")
    print(f"  {mnist_out}/")
    print(f"  {cifar_out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.show)
