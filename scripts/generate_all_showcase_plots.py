"""Genera todas las graficas posibles del proyecto con los checkpoints actuales.

Uso basico (procesa MNIST y CIFAR-10 con los checkpoints por defecto):
    python scripts/generate_all_showcase_plots.py

Solo MNIST:
    python scripts/generate_all_showcase_plots.py --dataset mnist

Solo CIFAR-10 con un checkpoint especifico (ej: al llegar a 100k pasos):
    python scripts/generate_all_showcase_plots.py --dataset cifar10 --checkpoint checkpoints/cifar10/step_0100000.pt

Con ventana de visualizacion ademas de guardar:
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
  [CIFAR-10 - checkpoint parcial o completo]
    08_samples_cifar10.png        muestras con el checkpoint indicado
    09_sample_evolution_cifar.png progresion desde paso 1k hasta el actual
"""
import sys
import os
import argparse

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
from scripts.plot_style import (
    apply_dark_style,
    BACKGROUND_DARK,
    AXES_BACKGROUND,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER_COLOR,
    COLOR_INDIGO,
    COLOR_BLUE,
    COLOR_AMBER,
    COLOR_EMERALD,
    COLOR_ROSE,
    COLOR_VIOLET,
    tensor_to_uint8_image,
)
from scripts import viz_diffusion, viz_ablation, viz_metrics

apply_dark_style()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_figure(figure, output_path, show_after_saving):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    print(f"  -> {output_path}")
    if show_after_saving:
        plt.show()
    else:
        plt.close(figure)


def load_model_from_config(config_path, checkpoint_path, device):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    model_config = config["model"]
    model = UNet(
        image_channels=config["dataset"]["image_channels"],
        base_channels=model_config["base_channels"],
        channel_multipliers=tuple(model_config["channel_multipliers"]),
        num_res_blocks=model_config["num_res_blocks"],
        attention_resolutions=tuple(model_config["attention_resolutions"]),
        dropout=model_config["dropout"],
        num_groups=model_config["num_groups"],
    ).to(device)
    ema = ExponentialMovingAverage.from_model(model)
    load_checkpoint(checkpoint_path, model, ema, device=device)
    ema.copy_to(model.parameters())
    model.eval()
    betas = make_linear_beta_schedule(
        config["diffusion"]["num_timesteps"],
        config["diffusion"]["beta_start"],
        config["diffusion"]["beta_end"],
    )
    diffusion = GaussianDiffusion(betas)
    return model, diffusion, config


def make_image_grid_figure(image_tensors, num_rows, num_cols, title, caption_list=None):
    num_images_to_show = min(len(image_tensors), num_rows * num_cols)
    figure, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.6, num_rows * 1.6 + 0.5))
    figure.patch.set_facecolor(BACKGROUND_DARK)
    axes_flat = axes.flatten() if num_rows * num_cols > 1 else [axes]
    for cell_idx in range(num_rows * num_cols):
        ax = axes_flat[cell_idx]
        ax.set_facecolor(BACKGROUND_DARK)
        ax.axis("off")
        if cell_idx < num_images_to_show:
            image_tensor = image_tensors[cell_idx]
            if image_tensor.shape[0] == 1:
                image_tensor = image_tensor.repeat(3, 1, 1)
            ax.imshow(tensor_to_uint8_image(image_tensor))
            if caption_list and cell_idx < len(caption_list):
                ax.set_title(caption_list[cell_idx], color=TEXT_SECONDARY, fontsize=7, pad=2)
    figure.suptitle(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.15)
    return figure


def plot_noise_schedules(output_dir, show_after_saving):
    print("\n[1/9] Comparativa de schedules de ruido...")
    ablation = NoiseScheduleAblation(1000)
    schedules_plot_data = {}
    for schedule_name in ("linear", "cosine", "sigmoid"):
        diffusion_obj = ablation.diffusion_objects[schedule_name]
        alpha_bar_values = diffusion_obj.alphas_cumprod.numpy()
        schedules_plot_data[schedule_name] = {
            "timesteps":      np.arange(1000),
            "betas":          diffusion_obj.betas.numpy(),
            "alphas_cumprod": alpha_bar_values,
            "snr":            alpha_bar_values / (1.0 - alpha_bar_values + 1e-8),
        }
    figure = viz_diffusion.plot_noise_schedule_overview(
        schedules_plot_data,
        title="Comparativa de Schedules de Ruido: beta, alpha_bar, SNR",
    )
    save_figure(figure, os.path.join(output_dir, "01_noise_schedules.png"), show_after_saving)


def plot_forward_process(dataset_name, num_image_channels, image_size_pixels, output_dir, show_after_saving):
    print(f"\n[2/9] Proceso forward sobre {dataset_name}...")
    from torchvision import datasets, transforms

    channel_mean_values = [0.5] * num_image_channels
    channel_std_values  = [0.5] * num_image_channels
    image_preprocessing_transform = transforms.Compose([
        transforms.Resize(image_size_pixels),
        transforms.ToTensor(),
        transforms.Normalize(channel_mean_values, channel_std_values),
    ])
    try:
        if dataset_name == "mnist":
            reference_dataset = datasets.MNIST(
                "data/raw", train=False, download=True, transform=image_preprocessing_transform
            )
        else:
            reference_dataset = datasets.CIFAR10(
                "data/raw", train=False, download=True, transform=image_preprocessing_transform
            )
        x_clean = reference_dataset[3][0]
    except Exception:
        x_clean = torch.randn(num_image_channels, image_size_pixels, image_size_pixels) * 0.3

    betas = make_linear_beta_schedule(1000, 1e-4, 0.02)
    diffusion = GaussianDiffusion(betas)
    timesteps_to_visualize = [0, 50, 100, 200, 400, 600, 800, 999]
    noisy_samples_by_timestep = {}
    fixed_noise = torch.randn_like(x_clean.unsqueeze(0))
    for timestep_value in timesteps_to_visualize[1:]:
        timestep_batch = torch.tensor([timestep_value])
        noisy_samples_by_timestep[timestep_value] = diffusion.q_sample(
            x_clean.unsqueeze(0), timestep_batch, fixed_noise
        )[0]

    figure = viz_diffusion.plot_forward_process_strip(
        x_clean,
        noisy_samples_by_timestep,
        title=f"Proceso Forward - {dataset_name.upper()} (schedule lineal)",
    )
    save_figure(figure, os.path.join(output_dir, f"02_forward_process_{dataset_name}.png"), show_after_saving)


def plot_samples_grid(model, diffusion, config, output_dir, dataset_tag, show_after_saving):
    print(f"\n[3/9] Cuadricula de muestras - {dataset_tag}...")
    num_image_channels = config["dataset"]["image_channels"]
    image_size_pixels  = config["dataset"]["image_size"]
    with torch.no_grad():
        generated_samples = diffusion.sample(model, 64, num_image_channels, image_size_pixels, DEVICE)
    sample_list = [generated_samples[i].cpu() for i in range(64)]
    figure = make_image_grid_figure(
        sample_list, 8, 8,
        f"64 muestras DDPM - {dataset_tag.upper()} (pesos EMA)",
    )
    save_figure(figure, os.path.join(output_dir, f"03_samples_grid_{dataset_tag}.png"), show_after_saving)


def plot_sample_evolution(config_path, checkpoint_dir, dataset_name, output_dir, show_after_saving):
    print(f"\n[4/9] Evolucion de calidad por paso - {dataset_name}...")
    all_step_checkpoints = sorted([
        filename for filename in os.listdir(checkpoint_dir)
        if filename.startswith("step_") and filename.endswith(".pt")
    ])
    if not all_step_checkpoints:
        print("  Sin checkpoints por paso disponibles.")
        return

    num_columns_to_show = min(6, len(all_step_checkpoints))
    evenly_spaced_indices = np.linspace(0, len(all_step_checkpoints) - 1, num_columns_to_show).astype(int)
    selected_checkpoint_files = [all_step_checkpoints[idx] for idx in evenly_spaced_indices]

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    model_config       = config["model"]
    num_image_channels = config["dataset"]["image_channels"]
    image_size_pixels  = config["dataset"]["image_size"]
    betas = make_linear_beta_schedule(
        config["diffusion"]["num_timesteps"],
        config["diffusion"]["beta_start"],
        config["diffusion"]["beta_end"],
    )
    diffusion = GaussianDiffusion(betas)

    figure, axes = plt.subplots(4, num_columns_to_show, figsize=(num_columns_to_show * 2.0, 9))
    figure.patch.set_facecolor(BACKGROUND_DARK)

    fixed_noise_for_comparison = torch.randn(
        4, num_image_channels, image_size_pixels, image_size_pixels, device=DEVICE
    )

    for col_idx, checkpoint_filename in enumerate(selected_checkpoint_files):
        step_number     = int(checkpoint_filename.replace("step_", "").replace(".pt", ""))
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        model_at_step = UNet(
            image_channels=num_image_channels,
            base_channels=model_config["base_channels"],
            channel_multipliers=tuple(model_config["channel_multipliers"]),
            num_res_blocks=model_config["num_res_blocks"],
            attention_resolutions=tuple(model_config["attention_resolutions"]),
            dropout=model_config["dropout"],
            num_groups=model_config["num_groups"],
        ).to(DEVICE)
        ema_at_step = ExponentialMovingAverage.from_model(model_at_step)
        load_checkpoint(checkpoint_path, model_at_step, ema_at_step, device=DEVICE)
        ema_at_step.copy_to(model_at_step.parameters())
        model_at_step.eval()

        with torch.no_grad():
            ddim_fast = DDIMSampler(diffusion, num_steps=50, eta=0.0)
            denoised_images = fixed_noise_for_comparison[:4].clone()
            for step_idx in reversed(range(len(ddim_fast.ddim_timesteps))):
                denoised_images = ddim_fast.ddim_step(model_at_step, denoised_images, step_idx)

        for row_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(BACKGROUND_DARK)
            ax.axis("off")
            image_tensor = denoised_images[row_idx].cpu()
            if image_tensor.shape[0] == 1:
                image_tensor = image_tensor.repeat(3, 1, 1)
            ax.imshow(tensor_to_uint8_image(image_tensor))
            if row_idx == 0:
                ax.set_title(f"paso\n{step_number:,}", color=COLOR_INDIGO, fontsize=9, fontweight="bold")

        del model_at_step, ema_at_step

    figure.suptitle(
        f"Evolucion de calidad - {dataset_name.upper()} (misma semilla, distintos checkpoints)",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=0.2)
    save_figure(figure, os.path.join(output_dir, f"04_sample_evolution_{dataset_name}.png"), show_after_saving)


def plot_reverse_chain(model, diffusion, config, output_dir, dataset_tag, show_after_saving):
    print(f"\n[5/9] Cadena inversa - {dataset_tag}...")
    num_image_channels = config["dataset"]["image_channels"]
    image_size_pixels  = config["dataset"]["image_size"]

    num_frames_to_save = 10
    save_at_timesteps = list(range(
        0, diffusion.num_timesteps, diffusion.num_timesteps // num_frames_to_save
    ))

    with torch.no_grad():
        _, saved_frames = diffusion.sample_progressive(
            model,
            batch_size=4,
            image_channels=num_image_channels,
            image_size=image_size_pixels,
            save_at_timesteps=save_at_timesteps,
            device=DEVICE,
        )

    frames_sorted_high_to_low_t = sorted(saved_frames, key=lambda frame_tuple: frame_tuple[0], reverse=True)

    num_columns = len(frames_sorted_high_to_low_t)
    figure, axes = plt.subplots(4, num_columns, figsize=(num_columns * 1.8, 8))
    figure.patch.set_facecolor(BACKGROUND_DARK)

    for col_idx, (timestep_value, batch_tensor) in enumerate(frames_sorted_high_to_low_t):
        for row_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(BACKGROUND_DARK)
            ax.axis("off")
            image_tensor = batch_tensor[row_idx]
            if image_tensor.shape[0] == 1:
                image_tensor = image_tensor.repeat(3, 1, 1)
            ax.imshow(tensor_to_uint8_image(image_tensor))
            if row_idx == 0:
                ax.set_title(f"t={timestep_value}", color=TEXT_SECONDARY, fontsize=8)

    figure.suptitle(
        f"Cadena Inversa - {dataset_tag.upper()}: ruido puro -> imagen generada",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=0.15)
    save_figure(figure, os.path.join(output_dir, f"05_reverse_chain_{dataset_tag}.png"), show_after_saving)


def plot_ddim_comparison(model, diffusion, config, output_dir, dataset_tag, show_after_saving):
    print(f"\n[6/9] DDIM vs DDPM - {dataset_tag}...")
    num_image_channels = config["dataset"]["image_channels"]
    image_size_pixels  = config["dataset"]["image_size"]

    step_configurations = [1000, 100, 50, 20, 10]
    timing_per_label    = {}
    samples_per_label   = {}

    fixed_noise_8_images = torch.randn(8, num_image_channels, image_size_pixels, image_size_pixels, device=DEVICE)

    for num_sampling_steps in step_configurations:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            if num_sampling_steps == 1000:
                generated_samples = diffusion.sample(model, 8, num_image_channels, image_size_pixels, DEVICE)
                run_label = "DDPM T=1000"
            else:
                ddim_sampler = DDIMSampler(diffusion, num_steps=num_sampling_steps, eta=0.0)
                generated_samples = ddim_sampler.sample(model, 8, num_image_channels, image_size_pixels, DEVICE)
                run_label = f"DDIM S={num_sampling_steps}"

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start_time
        timing_per_label[run_label] = elapsed_seconds
        samples_per_label[run_label] = [generated_samples[i].cpu() for i in range(8)]

    num_configurations = len(step_configurations)
    figure = plt.figure(figsize=(num_configurations * 2.2, 12))
    figure.patch.set_facecolor(BACKGROUND_DARK)

    for col_idx, (run_label, image_list) in enumerate(samples_per_label.items()):
        for row_idx in range(4):
            ax = figure.add_subplot(6, num_configurations, row_idx * num_configurations + col_idx + 1)
            ax.set_facecolor(BACKGROUND_DARK)
            ax.axis("off")
            image_tensor = image_list[row_idx]
            if image_tensor.shape[0] == 1:
                image_tensor = image_tensor.repeat(3, 1, 1)
            ax.imshow(tensor_to_uint8_image(image_tensor))
            if row_idx == 0:
                elapsed = timing_per_label[run_label]
                label_color = COLOR_BLUE if "DDIM" in run_label else COLOR_ROSE
                ax.set_title(f"{run_label}\n{elapsed:.1f}s", color=label_color, fontsize=8, fontweight="bold")

    ax_speedup_bars = figure.add_subplot(6, 1, 6)
    ax_speedup_bars.set_facecolor(AXES_BACKGROUND)
    bar_labels  = list(timing_per_label.keys())
    bar_times   = list(timing_per_label.values())
    bar_colors  = [COLOR_ROSE if "DDPM" in label else COLOR_BLUE for label in bar_labels]
    bars = ax_speedup_bars.bar(bar_labels, bar_times, color=bar_colors, alpha=0.85, edgecolor=BORDER_COLOR)
    for bar, elapsed in zip(bars, bar_times):
        ax_speedup_bars.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{elapsed:.1f}s", ha="center", color=TEXT_SECONDARY, fontsize=8,
        )
    ax_speedup_bars.set_ylabel("Segundos (8 imagenes)", color=TEXT_SECONDARY, fontsize=9)
    ax_speedup_bars.tick_params(colors=TEXT_SECONDARY)
    for spine in ax_speedup_bars.spines.values():
        spine.set_color(BORDER_COLOR)
    ax_speedup_bars.set_facecolor(AXES_BACKGROUND)
    baseline_ddpm_time = timing_per_label.get("DDPM T=1000", 1)
    for run_label, elapsed in timing_per_label.items():
        if "DDIM" in run_label:
            speedup_factor = baseline_ddpm_time / elapsed
            print(f"  {run_label}: {elapsed:.2f}s  ({speedup_factor:.1f}x speedup vs DDPM)")

    figure.suptitle(
        f"DDPM vs DDIM - {dataset_tag.upper()} (misma semilla, pesos EMA)",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold",
    )
    plt.tight_layout(pad=0.3)
    save_figure(figure, os.path.join(output_dir, f"06_ddim_vs_ddpm_{dataset_tag}.png"), show_after_saving)


def plot_interpolation(model, diffusion, config, output_dir, dataset_tag, show_after_saving):
    print(f"\n[7/9] Interpolacion en espacio latente - {dataset_tag}...")
    from torchvision import datasets, transforms

    num_image_channels = config["dataset"]["image_channels"]
    image_size_pixels  = config["dataset"]["image_size"]

    channel_mean_values = [0.5] * num_image_channels
    channel_std_values  = [0.5] * num_image_channels
    image_preprocessing_transform = transforms.Compose([
        transforms.Resize(image_size_pixels),
        transforms.ToTensor(),
        transforms.Normalize(channel_mean_values, channel_std_values),
    ])
    try:
        if "mnist" in config["dataset"]["name"]:
            reference_dataset = datasets.MNIST(
                "data/raw", train=False, download=True, transform=image_preprocessing_transform
            )
        else:
            reference_dataset = datasets.CIFAR10(
                "data/raw", train=False, download=True, transform=image_preprocessing_transform
            )
        first_reference_image  = reference_dataset[0][0].unsqueeze(0).to(DEVICE)
        second_reference_image = reference_dataset[7][0].unsqueeze(0).to(DEVICE)
    except Exception:
        first_reference_image  = torch.randn(1, num_image_channels, image_size_pixels, image_size_pixels, device=DEVICE)
        second_reference_image = torch.randn(1, num_image_channels, image_size_pixels, image_size_pixels, device=DEVICE)

    interpolator = LatentSpaceInterpolator(
        diffusion, interpolation_t=400, num_interpolation_steps=9
    )
    decoded_interpolation_images, lambda_values = interpolator.generate_interpolation_grid(
        model, first_reference_image, second_reference_image,
        noise_seed=42, device=DEVICE, ddim_steps=50,
    )

    num_interpolated_images = len(decoded_interpolation_images)
    figure, axes = plt.subplots(1, num_interpolated_images + 2, figsize=((num_interpolated_images + 2) * 2.0, 2.6))
    figure.patch.set_facecolor(BACKGROUND_DARK)

    panels_to_draw = (
        [(axes[0], "Original A", first_reference_image[0].cpu())]
        + [(axes[i + 1], f"lambda={lam:.2f}", decoded_interpolation_images[i][0])
           for i, lam in enumerate(lambda_values)]
        + [(axes[-1], "Original B", second_reference_image[0].cpu())]
    )
    for ax, panel_label, image_tensor in panels_to_draw:
        ax.set_facecolor(BACKGROUND_DARK)
        ax.axis("off")
        display_tensor = image_tensor if image_tensor.shape[0] != 1 else image_tensor.repeat(3, 1, 1)
        ax.imshow(tensor_to_uint8_image(display_tensor))
        ax.set_title(panel_label, color=COLOR_VIOLET, fontsize=7.5, fontweight="bold")

    figure.suptitle(
        f"Interpolacion en Espacio Latente - {dataset_tag.upper()} (t_interp=400, DDIM S=50)",
        color=TEXT_PRIMARY, fontsize=11, fontweight="bold", y=1.04,
    )
    plt.tight_layout(pad=0.2)
    save_figure(figure, os.path.join(output_dir, f"07_interpolation_{dataset_tag}.png"), show_after_saving)


def main(show_after_saving, target_dataset, override_checkpoint_path):
    mnist_config_path  = "configs/mnist.yaml"
    cifar_config_path  = "configs/cifar10.yaml"
    mnist_output_dir   = "checkpoints/mnist/plots"
    cifar_output_dir   = "checkpoints/cifar10/plots"

    # Checkpoint activo: --checkpoint sobreescribe el default.
    # Por defecto MNIST usa best.pt (ya convergio) y CIFAR-10 usa latest.pt (en progreso).
    if override_checkpoint_path and target_dataset != "both":
        mnist_checkpoint_path = override_checkpoint_path if target_dataset == "mnist"  else "checkpoints/mnist/best.pt"
        cifar_checkpoint_path = override_checkpoint_path if target_dataset == "cifar10" else "checkpoints/cifar10/latest.pt"
    else:
        mnist_checkpoint_path = "checkpoints/mnist/best.pt"
        cifar_checkpoint_path = "checkpoints/cifar10/latest.pt"

    run_mnist  = target_dataset in ("mnist",  "both")
    run_cifar  = target_dataset in ("cifar10", "both")

    no_model_output_dir = cifar_output_dir if run_cifar else mnist_output_dir
    plot_noise_schedules(no_model_output_dir, show_after_saving)

    if run_mnist:
        plot_forward_process("mnist", 1, 28, mnist_output_dir, show_after_saving)
    if run_cifar:
        plot_forward_process("cifar10", 3, 32, cifar_output_dir, show_after_saving)

    if run_mnist:
        if os.path.exists(mnist_checkpoint_path):
            model_mnist, diffusion_mnist, config_mnist = load_model_from_config(
                mnist_config_path, mnist_checkpoint_path, DEVICE
            )
            plot_samples_grid(model_mnist, diffusion_mnist, config_mnist, mnist_output_dir, "mnist", show_after_saving)
            plot_sample_evolution(mnist_config_path, "checkpoints/mnist", "mnist", mnist_output_dir, show_after_saving)
            plot_reverse_chain(model_mnist, diffusion_mnist, config_mnist, mnist_output_dir, "mnist", show_after_saving)
            plot_ddim_comparison(model_mnist, diffusion_mnist, config_mnist, mnist_output_dir, "mnist", show_after_saving)
            plot_interpolation(model_mnist, diffusion_mnist, config_mnist, mnist_output_dir, "mnist", show_after_saving)
            del model_mnist
        else:
            print(f"Sin checkpoint de MNIST en {mnist_checkpoint_path}, saltando graficas del modelo.")

    if run_cifar:
        if os.path.exists(cifar_checkpoint_path):
            model_cifar, diffusion_cifar, config_cifar = load_model_from_config(
                cifar_config_path, cifar_checkpoint_path, DEVICE
            )
            plot_samples_grid(model_cifar, diffusion_cifar, config_cifar, cifar_output_dir, "cifar10", show_after_saving)
            plot_sample_evolution(cifar_config_path, "checkpoints/cifar10", "cifar10", cifar_output_dir, show_after_saving)
            plot_reverse_chain(model_cifar, diffusion_cifar, config_cifar, cifar_output_dir, "cifar10", show_after_saving)
            del model_cifar
        else:
            print(f"Sin checkpoint de CIFAR-10 en {cifar_checkpoint_path}.")

    print("\nListo. Graficas guardadas en:")
    if run_mnist:
        print(f"  {mnist_output_dir}/")
    if run_cifar:
        print(f"  {cifar_output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera todas las graficas del proyecto DDPM.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar cada grafica en ventana ademas de guardarla.",
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10", "both"],
        default="both",
        help=(
            "Que dataset procesar. "
            "mnist: solo MNIST (modelo ya entrenado). "
            "cifar10: solo CIFAR-10 (checkpoint en progreso). "
            "both: ambos (default)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Ruta exacta al checkpoint que se quiere usar. "
            "Requiere --dataset mnist o --dataset cifar10 (no funciona con both). "
            "Ejemplos: "
            "checkpoints/cifar10/latest.pt, "
            "checkpoints/cifar10/step_0100000.pt, "
            "checkpoints/cifar10/best.pt"
        ),
    )
    args = parser.parse_args()

    if args.checkpoint and args.dataset == "both":
        print(
            "Advertencia: --checkpoint se ignora cuando --dataset es 'both'. "
            "Usa --dataset mnist o --dataset cifar10 para especificar un checkpoint concreto."
        )

    main(args.show, args.dataset, args.checkpoint)
