"""Demo interactiva con Gradio (Extra 1, Seccion 5.1 del briefing).

Caracteristicas:
  - Slider de pasos de muestreo: DDPM 1000 vs DDIM 50/20/10
  - Control de semilla para mostrar reproducibilidad en vivo
  - Visualizacion de la cadena inversa (como emerge la imagen del ruido)
  - Noise-then-denoise: sube una imagen, se corrompe y se reconstruye
  - Comparacion lado a lado DDPM vs DDIM a la misma semilla

Uso:
    python demo/app.py --checkpoint checkpoints/cifar10/best.pt --config configs/cifar10.yaml
    python demo/app.py --checkpoint checkpoints/mnist/best.pt   --config configs/mnist.yaml
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


PAGE_BACKGROUND    = "#f5f5f8"
CARD_BACKGROUND    = "#ffffff"
SUBTLE_BG          = "#f0f0f6"
BORDER_COLOR       = "#e4e4ee"
TEXT_PRIMARY       = "#0e0e1c"
TEXT_SECONDARY     = "#64649a"
TEXT_MUTED         = "#b4b4cc"
ACCENT_INDIGO      = "#5b5be8"
ACCENT_INDIGO_DARK = "#4848d0"
ACCENT_LIGHT       = "#eaeaff"
ACCENT_VIOLET      = "#7c3aed"

PANEL_BACKGROUND    = CARD_BACKGROUND
DARK_BACKGROUND     = PAGE_BACKGROUND
BORDER_COLOR_SUBTLE = BORDER_COLOR
ACCENT_ROSE         = "#e5484d"
ACCENT_EMERALD      = "#30a46c"


DEMO_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---- Base ---- */
.gradio-container {{
    background-color: {PAGE_BACKGROUND} !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-size: 15.5px !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 8px 12px !important;
    color: {TEXT_PRIMARY} !important;
}}

body, html {{
    background-color: {PAGE_BACKGROUND} !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ---- Hero: barra horizontal compacta ---- */
.ddpm-hero {{
    background: linear-gradient(110deg, #13112a 0%, #1d1840 55%, #16122e 100%);
    border-radius: 14px;
    padding: 20px 32px;
    margin-bottom: 14px;
    display: flex !important;
    align-items: center !important;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 8px 40px rgba(8,7,20,0.35),
        0 2px 8px rgba(30,24,64,0.25),
        inset 0 1px 0 rgba(255,255,255,0.06);
}}
.ddpm-hero::after {{
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(110,80,220,0.22) 0%, transparent 70%);
    pointer-events: none;
}}
.ddpm-hero .hero-eyebrow {{
    font-size: 0.66em !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: rgba(200,190,255,0.5) !important;
    margin: 0 0 6px 0 !important;
    display: block !important;
}}
.ddpm-hero h1 {{
    color: #f4f2ff !important;
    font-size: 1.6em !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.6px !important;
    line-height: 1.15 !important;
    text-shadow: 0 1px 12px rgba(0,0,0,0.25) !important;
    white-space: nowrap !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ---- Cards: solo primer nivel tiene sombra/borde ---- */
div[data-testid="block"] {{
    background: {CARD_BACKGROUND} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 14px !important;
    box-shadow:
        0 2px 16px rgba(40,40,120,0.07),
        0 1px 3px rgba(0,0,0,0.04),
        inset 0 1px 0 rgba(255,255,255,0.9) !important;
    transition: box-shadow 0.2s !important;
}}
div[data-testid="block"]:focus-within {{
    box-shadow:
        0 4px 24px rgba(91,91,232,0.12),
        0 1px 4px rgba(0,0,0,0.05),
        inset 0 1px 0 rgba(255,255,255,0.9) !important;
}}

/* Bloques anidados: sin card propia, se funden con el padre */
div[data-testid="block"] div[data-testid="block"] {{
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    border-radius: 0 !important;
    padding: 2px 0 !important;
}}

/* Reducir gap entre componentes dentro de columnas */
.gap {{ gap: 4px !important; }}

/* Reducir margen entre filas/columnas */
.row {{ gap: 8px !important; }}
.tabs {{ margin-top: 4px !important; }}

/* ---- Labels ---- */
label > span, .label-wrap > span, .label-wrap span {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.92em !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}}

/* Texto de info de sliders */
.info {{
    font-size: 0.83em !important;
    color: {TEXT_MUTED} !important;
    margin-top: 2px !important;
}}

/* ---- Inputs numericos ---- */
input[type="number"] {{
    background: {SUBTLE_BG} !important;
    color: {TEXT_PRIMARY} !important;
    font-size: 1em !important;
    font-weight: 600 !important;
    border: 1.5px solid {BORDER_COLOR} !important;
    border-radius: 8px !important;
}}
input[type="number"]:focus {{
    border-color: {ACCENT_INDIGO} !important;
    box-shadow: 0 0 0 3px {ACCENT_LIGHT} !important;
    outline: none !important;
}}

/* ---- Sliders ---- */
input[type="range"] {{ accent-color: {ACCENT_INDIGO} !important; }}

/* ---- Checkbox ---- */
input[type="checkbox"] {{ accent-color: {ACCENT_INDIGO} !important; }}
.checkbox-wrap label span {{
    font-size: 0.95em !important;
    font-weight: 600 !important;
    color: {TEXT_PRIMARY} !important;
}}

/* ---- Boton principal ---- */
button.primary, .gr-button-primary, button[variant="primary"] {{
    background: {ACCENT_INDIGO} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95em !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 15px 28px !important;
    box-shadow:
        0 4px 16px rgba(91,91,232,0.35),
        0 1px 4px rgba(91,91,232,0.2) !important;
    transition: all 0.18s !important;
    cursor: pointer !important;
}}
button.primary:hover {{
    background: {ACCENT_INDIGO_DARK} !important;
    transform: translateY(-1px) !important;
    box-shadow:
        0 8px 24px rgba(91,91,232,0.42),
        0 2px 6px rgba(91,91,232,0.25) !important;
}}
button.primary:active {{
    transform: translateY(0) !important;
}}

/* ---- Tabs ---- */
.tab-nav, div[role="tablist"] {{
    background: transparent !important;
    border-bottom: 2px solid {BORDER_COLOR} !important;
    margin-bottom: 12px !important;
}}
.tab-nav button, div[role="tab"] {{
    color: {TEXT_MUTED} !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    padding: 12px 24px !important;
    font-size: 1em !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: color 0.15s !important;
}}
.tab-nav button.selected, div[role="tab"][aria-selected="true"] {{
    color: {ACCENT_INDIGO} !important;
    border-bottom-color: {ACCENT_INDIGO} !important;
}}
.tab-nav button:hover:not(.selected) {{
    color: {TEXT_SECONDARY} !important;
}}

/* ---- Imagenes ---- */
.image-container, .output-image, div[data-testid="image"] {{
    background: {SUBTLE_BG} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 10px !important;
}}
div[data-testid="image"] img, .image-container img {{
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
    image-rendering: pixelated;
    border-radius: 6px !important;
}}

/* ---- Galeria ---- */
.gallery {{
    background: {SUBTLE_BG} !important;
    gap: 6px !important;
}}
.gallery > .thumbnail-item {{
    border-radius: 6px !important;
    border: 1px solid {BORDER_COLOR} !important;
}}

/* ---- Canvas de dibujo ---- */
div[data-testid="sketchpad"] {{
    background: {CARD_BACKGROUND} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 10px !important;
}}
div[data-testid="sketchpad"] canvas {{
    border-radius: 6px !important;
    border: 1.5px solid {ACCENT_INDIGO} !important;
    cursor: crosshair !important;
}}

/* ---- Texto markdown ---- */
.prose, .md, .prose p, .md p {{ color: {TEXT_SECONDARY} !important; }}
.prose strong, .md strong {{ color: {TEXT_PRIMARY} !important; }}
.prose a, .md a {{ color: {ACCENT_INDIGO} !important; text-decoration: none !important; }}

/* ---- Info card (descripcion de tabs) ---- */
.tab-info {{
    background: {ACCENT_LIGHT} !important;
    border-left: 4px solid {ACCENT_INDIGO} !important;
    border-radius: 0 10px 10px 0 !important;
    padding: 14px 20px !important;
    margin-bottom: 24px !important;
    font-size: 0.9em !important;
    color: {TEXT_SECONDARY} !important;
    line-height: 1.7 !important;
}}
.tab-info strong {{ color: {ACCENT_INDIGO} !important; font-weight: 700 !important; }}

/* ---- Galeria mas espaciosa ---- */
.gallery {{
    padding: 8px !important;
    gap: 8px !important;
    border-radius: 10px !important;
}}
.gallery > .thumbnail-item {{
    border-radius: 8px !important;
    border: 2px solid {BORDER_COLOR} !important;
    overflow: hidden !important;
    transition: border-color 0.15s !important;
}}
.gallery > .thumbnail-item:hover {{
    border-color: {ACCENT_INDIGO} !important;
}}

/* ---- Boton mas prominente ---- */
button.primary, .gr-button-primary, button[variant="primary"] {{
    padding: 14px 24px !important;
    font-size: 0.92em !important;
    border-radius: 10px !important;
    letter-spacing: 0.05em !important;
}}

/* ---- Imagen output sin borde visible ---- */
div[data-testid="image"] {{
    background: {SUBTLE_BG} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 12px !important;
    min-height: 100px !important;
}}
div[data-testid="image"] img {{
    border-radius: 8px !important;
    image-rendering: pixelated;
}}

/* ---- Shadows adicionales ---- */

/* Sombra sutil en el label de los sliders */
label > span, .label-wrap > span {{
    text-shadow: 0 1px 0 rgba(255,255,255,0.8) !important;
}}

/* Sombra en imagenes generadas para dar profundidad */
div[data-testid="image"] img {{
    filter: drop-shadow(0 4px 12px rgba(50,50,130,0.14)) !important;
}}

/* Sombra en la galeria */
.gallery > .thumbnail-item {{
    box-shadow: 0 2px 8px rgba(50,50,130,0.10) !important;
    transition: box-shadow 0.2s, transform 0.2s !important;
}}
.gallery > .thumbnail-item:hover {{
    box-shadow: 0 6px 20px rgba(91,91,232,0.20) !important;
    transform: translateY(-2px) !important;
}}

/* Sombra interior suave en inputs */
input[type="number"], input[type="text"] {{
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.06) !important;
}}

/* Sombra en el canvas de dibujo */
div[data-testid="sketchpad"] {{
    box-shadow: 0 4px 16px rgba(50,50,130,0.10), inset 0 1px 0 rgba(255,255,255,0.9) !important;
}}

/* Sombra en la barra de tabs */
.tab-nav, div[role="tablist"] {{
    box-shadow: 0 2px 0 {BORDER_COLOR} !important;
}}

/* ---- Scrollbars ---- */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {PAGE_BACKGROUND}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {TEXT_MUTED}; }}

footer {{ display: none !important; }}
"""


_model: UNet = None
_diffusion: GaussianDiffusion = None
_ema: ExponentialMovingAverage = None
_device: torch.device = None
_cfg: dict = None
_test_dataset = None  # cache del dataset para previews de interpolacion


MAX_PREVIEW_INDEX = 200  # cuantos indices exponer en los sliders


def load_model(config_path: str, checkpoint_path: str) -> None:
    global _model, _diffusion, _ema, _device, _cfg

    with open(config_path) as config_file:
        _cfg = yaml.safe_load(config_file)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = _cfg["model"]
    _model = UNet(
        image_channels=_cfg["dataset"]["image_channels"],
        base_channels=model_config["base_channels"],
        channel_multipliers=tuple(model_config["channel_multipliers"]),
        num_res_blocks=model_config["num_res_blocks"],
        attention_resolutions=tuple(model_config["attention_resolutions"]),
        dropout=model_config["dropout"],
        num_groups=model_config["num_groups"],
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
        print(f"Checkpoint cargado en paso {state['step']}")
    else:
        print("Checkpoint no encontrado. Usando pesos aleatorios (solo para demostrar la interfaz).")

    _model.eval()
    print(f"Demo lista en {_device}")


MIN_DISPLAY_SIZE_PIXELS = 256


def tensor_to_pil(tensor_image: torch.Tensor) -> Image.Image:
    """Convierte tensor (C,H,W) en [-1,1] a PIL Image.

    Escala a MIN_DISPLAY_SIZE_PIXELS si la imagen es mas pequena (ej. MNIST 28x28 -> 256x256).
    Usa nearest-neighbor para preservar el aspecto pixelado de los digits.
    """
    img = tensor_image.detach().cpu().float()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img_numpy = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if img_numpy.shape[2] == 1:
        img_numpy = np.repeat(img_numpy, 3, axis=2)
    pil_image = Image.fromarray(img_numpy)
    if pil_image.width < MIN_DISPLAY_SIZE_PIXELS or pil_image.height < MIN_DISPLAY_SIZE_PIXELS:
        pil_image = pil_image.resize(
            (MIN_DISPLAY_SIZE_PIXELS, MIN_DISPLAY_SIZE_PIXELS),
            Image.NEAREST,
        )
    return pil_image


@torch.no_grad()
def generate_image(
    num_sampling_steps: int,
    seed: int,
    use_ddim: bool,
    eta: float,
) -> Image.Image:
    """Genera una imagen con el muestreador seleccionado."""
    generator = torch.Generator(device=_device).manual_seed(int(seed))

    image_size     = _cfg["dataset"]["image_size"]
    image_channels = _cfg["dataset"]["image_channels"]

    if use_ddim:
        ddim_sampler = DDIMSampler(_diffusion, num_steps=int(num_sampling_steps), eta=eta)
        x_generated = ddim_sampler.sample(
            _model,
            batch_size=1,
            image_channels=image_channels,
            image_size=image_size,
            device=str(_device),
            generator=generator,
        )
    else:
        x_generated = _diffusion.sample(
            _model,
            batch_size=1,
            image_channels=image_channels,
            image_size=image_size,
            device=str(_device),
            generator=generator,
        )

    return tensor_to_pil(x_generated[0])


@torch.no_grad()
def generate_progressive_frames(num_sampling_steps: int, seed: int) -> list:
    """Genera frames del proceso inverso para visualizacion progresiva."""
    image_size     = _cfg["dataset"]["image_size"]
    image_channels = _cfg["dataset"]["image_channels"]
    num_frames_to_capture = min(12, num_sampling_steps)
    save_at_timesteps = list(range(
        0, _diffusion.num_timesteps, _diffusion.num_timesteps // num_frames_to_capture
    ))

    seed_everything(int(seed))
    _, saved_frames = _diffusion.sample_progressive(
        _model,
        batch_size=1,
        image_channels=image_channels,
        image_size=image_size,
        save_at_timesteps=save_at_timesteps,
        device=str(_device),
    )

    return [tensor_to_pil(frame_tensor[0]) for _, frame_tensor in saved_frames]


def sketchpad_to_pil(sketchpad_output) -> Image.Image:
    """Convierte el output de gr.Sketchpad a PIL Image.

    Gradio puede devolver dict (4.x reciente), numpy array, o PIL Image
    dependiendo de la version y configuracion del componente.
    """
    if sketchpad_output is None:
        return None

    if isinstance(sketchpad_output, dict):
        raw = (
            sketchpad_output.get("composite")
            or sketchpad_output.get("background")
            or sketchpad_output.get("image")
        )
        if raw is None:
            return None
        if isinstance(raw, np.ndarray):
            return Image.fromarray(raw.astype(np.uint8))
        return raw

    if isinstance(sketchpad_output, np.ndarray):
        return Image.fromarray(sketchpad_output.astype(np.uint8))

    return sketchpad_output


@torch.no_grad()
def noise_then_denoise_from_canvas(
    canvas_drawing,
    noise_level_t: int,
    seed: int,
) -> tuple:
    """Corrompe el dibujo del canvas con q_sample (Ec. 4) y lo reconstruye.

    El canvas devuelve trazos oscuros sobre fondo blanco. Para MNIST
    (digitos blancos sobre negro) invertimos la imagen antes de procesarla.
    """
    import torchvision.transforms as T

    pil_image = sketchpad_to_pil(canvas_drawing)
    if pil_image is None:
        return None, None

    image_size          = _cfg["dataset"]["image_size"]
    num_image_channels  = _cfg["dataset"]["image_channels"]

    if num_image_channels == 1:
        grayscale_image = pil_image.convert("L")
        # El canvas tiene trazo oscuro sobre blanco; MNIST es blanco sobre negro.
        # Invertimos para que el trazo del usuario sea la "senal" del modelo.
        from PIL import ImageOps
        grayscale_image = ImageOps.invert(grayscale_image)
        normalization   = T.Normalize([0.5], [0.5])
        input_image     = grayscale_image
    else:
        input_image   = pil_image.convert("RGB")
        normalization = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    image_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalization,
    ])
    x_start = image_transform(input_image).unsqueeze(0).to(_device)

    generator       = torch.Generator(device=_device).manual_seed(int(seed))
    noise           = torch.randn(x_start.shape, device=_device, generator=generator)
    timestep_batch  = torch.tensor([noise_level_t], device=_device, dtype=torch.long)
    x_noisy         = _diffusion.q_sample(x_start, timestep_batch, noise)

    x_reconstructed = x_noisy.clone()
    for t in reversed(range(noise_level_t)):
        x_reconstructed = _diffusion.reverse_step(_model, x_reconstructed, t)

    return tensor_to_pil(x_noisy[0]), tensor_to_pil(x_reconstructed[0])


def get_test_dataset():
    """Carga (y cachea) el test set del dataset configurado."""
    global _test_dataset
    if _test_dataset is not None:
        return _test_dataset

    from torchvision import datasets, transforms

    num_channels = _cfg["dataset"]["image_channels"]
    image_size   = _cfg["dataset"]["image_size"]
    channel_means = [0.5] * num_channels
    channel_stds  = [0.5] * num_channels

    preprocessing = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(channel_means, channel_stds),
    ])

    dataset_name = _cfg["dataset"]["name"]
    if "mnist" in dataset_name:
        _test_dataset = datasets.MNIST("data/raw", train=False, download=True, transform=preprocessing)
    else:
        _test_dataset = datasets.CIFAR10("data/raw", train=False, download=True, transform=preprocessing)

    return _test_dataset


def get_preview_image_at_index(image_index: int) -> Image.Image:
    """Devuelve la imagen del test set en image_index como PIL para preview."""
    dataset        = get_test_dataset()
    safe_index     = max(0, min(int(image_index), len(dataset) - 1))
    image_tensor, _ = dataset[safe_index]
    return tensor_to_pil(image_tensor)


@torch.no_grad()
def generate_latent_interpolation(
    image_index_start: int,
    image_index_end: int,
    interpolation_t: int,
) -> list:
    """Interpolacion lineal en espacio latente (Seccion 4.4 del paper).

    Codifica ambas imagenes al espacio ruidoso x_t con q_sample (Ec. 4),
    interpola linealmente, y decodifica cada punto con DDIM (50 pasos).
    Devuelve la tira completa: imagen A + 9 puntos interpolados + imagen B.
    """
    from extras.latent_interpolation import LatentSpaceInterpolator

    dataset = get_test_dataset()
    num_images_in_dataset = len(dataset)

    safe_index_start = max(0, min(int(image_index_start), num_images_in_dataset - 1))
    safe_index_end   = max(0, min(int(image_index_end),   num_images_in_dataset - 1))

    x_image_start = dataset[safe_index_start][0].unsqueeze(0).to(_device)
    x_image_end   = dataset[safe_index_end][0].unsqueeze(0).to(_device)

    interpolator = LatentSpaceInterpolator(
        _diffusion,
        interpolation_t=int(interpolation_t),
        num_interpolation_steps=9,
    )
    decoded_images, lambda_values = interpolator.generate_interpolation_grid(
        _model,
        x_image_start,
        x_image_end,
        noise_seed=42,
        device=str(_device),
        ddim_steps=50,
    )

    interpolation_strip = (
        [tensor_to_pil(x_image_start[0].cpu())]
        + [tensor_to_pil(interpolated_image[0]) for interpolated_image in decoded_images]
        + [tensor_to_pil(x_image_end[0].cpu())]
    )
    return interpolation_strip


def build_gradio_interface() -> gr.Blocks:
    light_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill=PAGE_BACKGROUND,
        block_background_fill=CARD_BACKGROUND,
        block_border_color=BORDER_COLOR,
        block_label_background_fill=CARD_BACKGROUND,
        block_label_text_color=TEXT_SECONDARY,
        body_text_color=TEXT_PRIMARY,
        body_text_color_subdued=TEXT_SECONDARY,
        input_background_fill=SUBTLE_BG,
        input_border_color=BORDER_COLOR,
        button_primary_background_fill=ACCENT_INDIGO,
        button_primary_background_fill_hover=ACCENT_INDIGO_DARK,
        button_primary_text_color="#ffffff",
        border_color_accent=ACCENT_INDIGO,
        color_accent=ACCENT_INDIGO,
        color_accent_soft=ACCENT_LIGHT,
        link_text_color=ACCENT_INDIGO,
        panel_background_fill=CARD_BACKGROUND,
        shadow_drop="0 1px 6px rgba(80,80,180,0.07)",
        shadow_spread="0px",
    )

    with gr.Blocks(theme=light_theme, css=DEMO_CSS, title="DDPM - Demo Interactiva") as demo_interface:

        gr.HTML("""
        <div class="ddpm-hero">
            <div>
                <span class="hero-eyebrow">NeurIPS 2020 &nbsp;&middot;&nbsp; Ho, Jain &amp; Abbeel &nbsp;&middot;&nbsp; Proyecto Final &nbsp;&middot;&nbsp; Reimplementación en PyTorch</span>
                <h1>Denoising Diffusion Probabilistic Models</h1>
            </div>
        </div>
        """)

        with gr.Tabs():

            with gr.Tab("Generación"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        use_ddim_toggle = gr.Checkbox(
                            label="Usar DDIM  (determinista, más rápido)",
                            value=True,
                        )
                        num_steps_slider = gr.Slider(
                            minimum=10, maximum=1000, value=50, step=10,
                            label="Pasos de muestreo",
                            info="50 pasos DDIM da calidad comparable a 1 000 pasos DDPM",
                        )
                        eta_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                            label="Eta  (0 = determinístico  /  1 = estocástico)",
                            info="eta=0: misma semilla siempre produce la misma imagen",
                        )
                        with gr.Row():
                            seed_input = gr.Number(value=42, label="Semilla", precision=0)
                        generate_btn = gr.Button("Generar imagen", variant="primary")

                    with gr.Column(scale=3):
                        output_image = gr.Image(
                            label="Imagen generada",
                            height=340,
                            show_label=False,
                        )

                generate_btn.click(
                    fn=generate_image,
                    inputs=[num_steps_slider, seed_input, use_ddim_toggle, eta_slider],
                    outputs=output_image,
                )

            with gr.Tab("Cadena Inversa"):
                gr.HTML("""
                <div class="tab-info">
                    Cada frame es un snapshot del proceso inverso.
                    El modelo parte de ruido gaussiano puro y elimina ruido paso a paso
                    hasta construir una imagen coherente.
                </div>
                """)
                with gr.Row():
                    progressive_steps_slider = gr.Slider(
                        10, 1000, value=200, step=10,
                        label="Pasos totales",
                    )
                    progressive_seed_input = gr.Number(
                        value=0, label="Semilla", precision=0,
                    )
                    progressive_btn = gr.Button(
                        "Generar cadena inversa", variant="primary", scale=2,
                    )
                progressive_gallery = gr.Gallery(
                    label="",
                    columns=6,
                    height=240,
                    object_fit="contain",
                    show_label=False,
                )
                progressive_btn.click(
                    fn=generate_progressive_frames,
                    inputs=[progressive_steps_slider, progressive_seed_input],
                    outputs=progressive_gallery,
                )

            with gr.Tab("Noise - Denoise"):
                gr.HTML("""
                <div class="tab-info">
                    <strong>Ecuación 4 del paper en vivo:</strong>
                    dibuja un dígito, corrómpe con q_sample hasta el paso t,
                    y observa cómo el proceso inverso lo reconstruye.
                    Trazo grueso para mejores resultados con MNIST.
                </div>
                """)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        draw_canvas = gr.Sketchpad(
                            label="Canvas",
                            type="pil",
                            height=260,
                        )
                        with gr.Row():
                            noise_level_t_slider = gr.Slider(
                                50, 950, value=400, step=50,
                                label="Nivel de ruido t",
                                info="t=400 destrucción moderada. t=800 casi ruido puro.",
                            )
                            nd_seed_input = gr.Number(
                                value=42, label="Semilla", precision=0,
                            )
                        nd_btn = gr.Button("Corromper y reconstruir", variant="primary")

                    with gr.Column(scale=2):
                        noisy_output = gr.Image(
                            label="Corrompida  x_t",
                            height=260,
                        )

                    with gr.Column(scale=2):
                        reconstructed_output = gr.Image(
                            label="Reconstruida",
                            height=260,
                        )

                nd_btn.click(
                    fn=noise_then_denoise_from_canvas,
                    inputs=[draw_canvas, noise_level_t_slider, nd_seed_input],
                    outputs=[noisy_output, reconstructed_output],
                )

            with gr.Tab("Interpolación Latente"):
                gr.HTML("""
                <div class="tab-info">
                    <strong>Sección 4.4 del paper:</strong>
                    las dos imágenes se codifican al espacio ruidoso en el paso t con q_sample,
                    se interpola linealmente, y cada punto se decodifica con DDIM.
                    <strong>t alto = transición suave. &nbsp; t bajo = transición abrupta.</strong>
                </div>
                """)

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        index_start_slider = gr.Slider(
                            0, MAX_PREVIEW_INDEX - 1, value=0, step=1,
                            label="Imagen A  (índice en el test set)",
                        )
                        preview_image_start = gr.Image(
                            label="",
                            height=170,
                            interactive=False,
                            show_label=False,
                        )

                    with gr.Column(scale=1):
                        index_end_slider = gr.Slider(
                            0, MAX_PREVIEW_INDEX - 1, value=7, step=1,
                            label="Imagen B  (índice en el test set)",
                        )
                        preview_image_end = gr.Image(
                            label="",
                            height=170,
                            interactive=False,
                            show_label=False,
                        )

                    with gr.Column(scale=1):
                        interpolation_t_slider = gr.Slider(
                            100, 900, value=400, step=50,
                            label="t de interpolación",
                            info="t=400 transición suave. t=700 muy suave.",
                        )
                        gr.HTML("<div style='height:20px'></div>")
                        interp_btn = gr.Button("Generar interpolación", variant="primary")

                interpolation_gallery = gr.Gallery(
                    label="A  →  interpolación lineal  →  B",
                    columns=11,
                    height=200,
                    object_fit="contain",
                )

                index_start_slider.change(
                    fn=get_preview_image_at_index,
                    inputs=[index_start_slider],
                    outputs=[preview_image_start],
                )
                index_end_slider.change(
                    fn=get_preview_image_at_index,
                    inputs=[index_end_slider],
                    outputs=[preview_image_end],
                )
                interp_btn.click(
                    fn=generate_latent_interpolation,
                    inputs=[index_start_slider, index_end_slider, interpolation_t_slider],
                    outputs=[interpolation_gallery],
                )

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
