"""Demo interactiva con Gradio (Extra 1, Seccion 5.1 del briefing).

Caracteristicas:
  - Slider de pasos de muestreo: DDPM 1000 vs DDIM 50/20/10
  - Control de semilla para mostrar reproducibilidad en vivo
  - Visualizacion de la cadena inversa (como emerge la imagen del ruido)
  - Noise-then-denoise: sube una imagen, se corrompe y se reconstruye
  - Comparacion lado a lado DDPM vs DDIM a la misma semilla

Uso:
    python demo/app.py --checkpoint checkpoints/cifar10/best.pt --config configs/cifar10.yaml
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import gradio as gr

from ddpm import GaussianDiffusion, UNet, ExponentialMovingAverage, DDIMSampler
from ddpm.diffusion import make_linear_beta_schedule
from utils import seed_everything
from utils.checkpointing import load_checkpoint

import yaml


# ---------------------------------------------------------------------------
# Estado global del modelo (se carga una vez al iniciar)
# ---------------------------------------------------------------------------

_model: UNet = None
_diffusion: GaussianDiffusion = None
_ema: ExponentialMovingAverage = None
_device: torch.device = None
_cfg: dict = None


def load_model(config_path: str, checkpoint_path: str) -> None:
    global _model, _diffusion, _ema, _device, _cfg

    with open(config_path) as f:
        _cfg = yaml.safe_load(f)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = _cfg["model"]
    _model = UNet(
        image_channels=_cfg["dataset"]["image_channels"],
        base_channels=model_cfg["base_channels"],
        channel_multipliers=tuple(model_cfg["channel_multipliers"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        attention_resolutions=tuple(model_cfg["attention_resolutions"]),
        dropout=model_cfg["dropout"],
        num_groups=model_cfg["num_groups"],
    ).to(_device)

    betas = make_linear_beta_schedule(
        _cfg["diffusion"]["num_timesteps"],
        _cfg["diffusion"]["beta_start"],
        _cfg["diffusion"]["beta_end"],
    )
    _diffusion = GaussianDiffusion(betas)
    _ema = ExponentialMovingAverage.from_model(_model)

    if os.path.exists(checkpoint_path):
        state = load_checkpoint(checkpoint_path, _model, _ema, device=str(_device))
        _ema.copy_to(_model.parameters())
        print(f"✅ Checkpoint cargado (paso {state['step']})")
    else:
        print("⚠️  Checkpoint no encontrado, usando pesos aleatorios (solo para demostrar interfaz)")

    _model.eval()
    print(f"🚀 Demo lista en {_device}")


def tensor_to_pil(tensor_image: torch.Tensor) -> Image.Image:
    """Convierte tensor (C,H,W) en [-1,1] a PIL Image."""
    img = tensor_image.detach().cpu().float()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    return Image.fromarray(img_np)


# ---------------------------------------------------------------------------
# Funciones de inferencia
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_image(
    num_sampling_steps: int,
    seed: int,
    use_ddim: bool,
    eta: float,
) -> Image.Image:
    """Genera una imagen con el muestreador seleccionado."""
    generator = torch.Generator(device=_device).manual_seed(int(seed))

    image_size = _cfg["dataset"]["image_size"]
    image_channels = _cfg["dataset"]["image_channels"]

    if use_ddim:
        ddim_sampler = DDIMSampler(_diffusion, num_steps=int(num_sampling_steps), eta=eta)
        x_generated = ddim_sampler.sample(
            _model, batch_size=1, image_channels=image_channels,
            image_size=image_size, device=str(_device), generator=generator,
        )
    else:
        x_generated = _diffusion.sample(
            _model, batch_size=1, image_channels=image_channels,
            image_size=image_size, device=str(_device), generator=generator,
        )

    return tensor_to_pil(x_generated[0])


@torch.no_grad()
def generate_progressive_gif(num_sampling_steps: int, seed: int) -> list:
    """Genera frames del proceso inverso para visualizacion progresiva."""
    image_size = _cfg["dataset"]["image_size"]
    image_channels = _cfg["dataset"]["image_channels"]
    num_frames_to_save = min(12, num_sampling_steps)
    save_at = list(range(0, _diffusion.num_timesteps, _diffusion.num_timesteps // num_frames_to_save))

    seed_everything(int(seed))
    _, saved_frames = _diffusion.sample_progressive(
        _model, batch_size=1,
        image_channels=image_channels, image_size=image_size,
        save_at_timesteps=save_at, device=str(_device),
    )

    pil_frames = []
    for _, frame_tensor in saved_frames:
        pil_frames.append(tensor_to_pil(frame_tensor[0]))

    return pil_frames


@torch.no_grad()
def noise_then_denoise(
    uploaded_image: Image.Image,
    noise_level_t: int,
    seed: int,
) -> tuple:
    """Corrompe una imagen real con el proceso forward y la reconstruye."""
    import torchvision.transforms as T

    image_size = _cfg["dataset"]["image_size"]

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]) if _cfg["dataset"]["image_channels"] == 1
        else T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    if _cfg["dataset"]["image_channels"] == 1 and uploaded_image.mode != "L":
        uploaded_image = uploaded_image.convert("L")
    elif _cfg["dataset"]["image_channels"] == 3 and uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")

    x_start = transform(uploaded_image).unsqueeze(0).to(_device)

    generator = torch.Generator(device=_device).manual_seed(int(seed))
    noise = torch.randn(x_start.shape, device=_device, generator=generator)
    timesteps_batch = torch.tensor([noise_level_t], device=_device, dtype=torch.long)
    x_noisy = _diffusion.q_sample(x_start, timesteps_batch, noise)

    # Reconstruir desde el paso noise_level_t hacia atras
    x_reconstructed = x_noisy.clone()
    for t in reversed(range(noise_level_t)):
        x_reconstructed = _diffusion.reverse_step(_model, x_reconstructed, t)

    noisy_pil = tensor_to_pil(x_noisy[0])
    reconstructed_pil = tensor_to_pil(x_reconstructed[0])
    return noisy_pil, reconstructed_pil


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------

def build_gradio_interface() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="indigo",
            neutral_hue="slate",
        ),
        title="DDPM — Demo Interactiva",
        css="""
        .dark-header { background: linear-gradient(135deg, #1e1b4b, #312e81);
                       padding: 20px; border-radius: 8px; margin-bottom: 16px; }
        .dark-header h1 { color: #e0e7ff; margin: 0; font-size: 1.6em; }
        .dark-header p  { color: #a5b4fc; margin: 4px 0 0 0; }
        """,
    ) as demo_interface:

        gr.HTML("""
        <div class="dark-header">
          <h1>⚙️ DDPM desde Cero — Demo Interactiva</h1>
          <p>Reimplementacion de Ho, Jain &amp; Abbeel, NeurIPS 2020 · PyTorch · Proyecto Final DL</p>
        </div>
        """)

        with gr.Tabs():
            # ---- Tab 1: Generacion ----
            with gr.Tab("🎨 Generacion"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Configuracion del muestreador")
                        use_ddim_toggle = gr.Checkbox(label="Usar DDIM (mas rapido)", value=True)
                        num_steps_slider = gr.Slider(
                            minimum=10, maximum=1000, value=50, step=10,
                            label="Pasos de muestreo",
                        )
                        eta_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                            label="Eta (0=deterministico, 1=DDPM)",
                            visible=True,
                        )
                        seed_input = gr.Number(value=42, label="Semilla", precision=0)
                        generate_btn = gr.Button("⚡ Generar", variant="primary")

                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Imagen generada", height=256)
                        gr.Markdown("""
                        **Tip:** Cambia la semilla y observa diferentes muestras.
                        Con eta=0 y DDIM, la misma semilla siempre produce la misma imagen.
                        """)

                generate_btn.click(
                    fn=generate_image,
                    inputs=[num_steps_slider, seed_input, use_ddim_toggle, eta_slider],
                    outputs=output_image,
                )

            # ---- Tab 2: Proceso inverso ----
            with gr.Tab("🔬 Cadena Inversa"):
                gr.Markdown("### Visualiza como el ruido se convierte en imagen paso a paso")
                with gr.Row():
                    progressive_steps_slider = gr.Slider(10, 1000, value=100, step=10, label="Pasos")
                    progressive_seed_input = gr.Number(value=0, label="Semilla", precision=0)
                progressive_btn = gr.Button("🎞️ Generar frames", variant="primary")
                progressive_gallery = gr.Gallery(
                    label="Frames del proceso inverso (izq=ruido → der=imagen)",
                    columns=6, height=200,
                )
                progressive_btn.click(
                    fn=generate_progressive_gif,
                    inputs=[progressive_steps_slider, progressive_seed_input],
                    outputs=progressive_gallery,
                )

            # ---- Tab 3: Noise-then-denoise ----
            with gr.Tab("🔊 Noise → Denoise"):
                gr.Markdown("### Corrompe una imagen real y reconstrúyela")
                with gr.Row():
                    with gr.Column():
                        uploaded_img = gr.Image(label="Imagen de entrada", type="pil", height=200)
                        noise_level_t_slider = gr.Slider(
                            50, 950, value=500, step=50,
                            label="Nivel de ruido t (mayor = mas destruccion)",
                        )
                        nd_seed_input = gr.Number(value=42, label="Semilla", precision=0)
                        nd_btn = gr.Button("🔄 Corromper y reconstruir", variant="primary")
                    with gr.Column():
                        noisy_output = gr.Image(label="Imagen corrompida x_t", height=200)
                        reconstructed_output = gr.Image(label="Imagen reconstruida", height=200)

                nd_btn.click(
                    fn=noise_then_denoise,
                    inputs=[uploaded_img, noise_level_t_slider, nd_seed_input],
                    outputs=[noisy_output, reconstructed_output],
                )

        gr.Markdown("""
        ---
        **Referencia:** Ho, Jain & Abbeel (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
        Repositorio oficial: [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
        """)

    return demo_interface


def main():
    parser = argparse.ArgumentParser(description="Demo interactiva DDPM")
    parser.add_argument("--config",     type=str, default="configs/cifar10.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cifar10/best.pt")
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--share",      action="store_true", help="Crear URL publica temporal")
    args = parser.parse_args()

    load_model(args.config, args.checkpoint)
    demo = build_gradio_interface()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
