<div align="center">

<h1>Denoising Diffusion Probabilistic Models</h1>
<h3>PyTorch reimplementation from scratch</h3>

<p>
  <a href="https://arxiv.org/abs/2006.11239">
    <img src="https://img.shields.io/badge/paper-arXiv%3A2006.11239-b31b1b?style=flat-square" alt="Paper"/>
  </a>
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"/>
</p>

<p>
  Reimplementación completa en PyTorch del paper <em>"Denoising Diffusion Probabilistic Models"</em><br/>
  Ho, Jain & Abbeel · NeurIPS 2020 · arXiv:2006.11239<br/><br/>
  <strong>Proyecto Final — Aprendizaje Profundo · Licenciatura en Ciencia de Datos · Junio 2026</strong>
</p>

</div>

---

## Contenido

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Extras implementados](#extras-implementados)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación y uso](#instalación-y-uso)
- [Resultados](#resultados)
- [Referencias](#referencias)

---

## Descripción

Este repositorio contiene una implementación completa de DDPM desde cero en PyTorch, sin usar módulos preentrenados ni arquitecturas importadas de librerías como `diffusers`. El objetivo fue entender cada componente del proceso de difusión, no solo hacerlo funcionar.

### El proceso de difusión

DDPM define un **proceso forward** que corrompe una imagen añadiendo ruido gaussiano en $T$ pasos, y un **proceso inverso** donde una red neuronal aprende a deshacer esa corrupción paso a paso.

La propiedad clave del proceso forward es que se puede saltar directamente al paso $t$ sin simular todos los intermedios:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\, x_0,\; (1 - \bar\alpha_t)\, \mathbf{I}\right)$$

Lo que se traduce directamente en:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\; \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I})$$

En lugar de pedir a la red que prediga la imagen limpia (lo intuitivo), Ho et al. demostraron que es más estable pedirle que prediga el ruido $\varepsilon$. La función de pérdida resultante es simplemente un MSE:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\, x_0,\, \varepsilon}\!\left[\left\lVert \varepsilon - \varepsilon_\theta\!\left(x_t,\, t\right) \right\rVert^2\right]$$

Para generar, se invierte el proceso paso a paso (Algoritmo 2 del paper):

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\right) + \sigma_t\, z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$

---

## Arquitectura

La red $\varepsilon_\theta$ es un **U-Net** con conexiones de salto entre niveles de igual resolución. Implementamos cada componente manualmente:

### Configuración CIFAR-10 (fiel al paper)

| Hiperparámetro | Valor |
|---|---|
| Timesteps $T$ | 1 000 |
| Schedule $\beta$ | Lineal, de $10^{-4}$ a $0.02$ |
| Canales base (`ch`) | 128 |
| Multiplicadores (`ch_mult`) | `(1, 2, 2, 2)` |
| Bloques residuales por nivel | 2 |
| Atención en resolución | 16 × 16 |
| Dropout | 0.1 |
| Optimizador | Adam, lr = 2e-4, batch = 128 |
| EMA decay | 0.9999 |
| Parámetros totales | ~35.7 M |

### Componentes implementados

**`SinusoidalTimestepEmbedding`** — embedding posicional de $t$ idéntico al de los Transformers originales. Convierte el timestep escalar en un vector de dimensión `ch`, que luego pasa por un MLP de dos capas antes de inyectarse en cada bloque residual.

**`ResidualBlock`** — GroupNorm → SiLU → Conv → *[+ time projection]* → GroupNorm → SiLU → Dropout → Conv, con conexión residual. La proyección del tiempo se suma a las features intermedias, condicionando el bloque en qué paso del proceso se encuentra.

**`SelfAttentionBlock`** — aplana la feature map a una secuencia de tokens espaciales y aplica `nn.MultiheadAttention`. Solo se activa en la resolución de 16 × 16, donde las dependencias de largo alcance son relevantes sin un costo computacional prohibitivo.

**`Downsample` / `Upsample`** — stride-2 conv para bajar resolución; nearest-neighbor interpolation seguido de conv para subir. Aprendemos el submuestreo en lugar de usar pooling fijo.

---

## Extras implementados

### 1. DDIM — Denoising Diffusion Implicit Models

Implementamos el muestreador DDIM (Song et al., 2021) sobre los pesos del modelo DDPM ya entrenado, sin reentrenar. El proceso de muestreo es determinista con `eta=0` y permite reducir los pasos de 1 000 a 50–100 con pérdida mínima de calidad:

| Muestreador | Pasos | FID (referencia) | Speedup |
|---|---|---|---|
| DDPM | 1 000 | ~3.17 | 1× |
| DDIM | 250 | ~4.2 | ~4× |
| DDIM | 100 | ~4.5 | ~10× |
| DDIM | 50 | ~4.8 | ~20× |
| DDIM | 20 | ~6.8 | ~50× |

### 2. Interpolación en espacio latente

Reproducimos la Sección 4.4 del paper. Codificamos dos imágenes al espacio ruidoso en el paso $t$ con `q_sample`, interpolamos linealmente, y decodificamos con el proceso inverso:

$$\bar{x}_t(\lambda) = (1 - \lambda)\, x_t + \lambda\, x_t', \qquad \bar{x}_0 \sim p_\theta(x_0 \mid \bar{x}_t(\lambda))$$

El parámetro $t$ de interpolación controla la suavidad de la transición: valores altos de $t$ producen mezclas más suaves porque más información de la imagen original se ha destruido antes de interpolar.

### 3. Ablación de schedules de ruido

Comparamos tres schedules:

- **Lineal** (paper original): $\beta_t$ crece linealmente de $10^{-4}$ a $0.02$
- **Coseno** (Nichol & Dhariwal, 2021): usa $\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$, evitando el colapso de SNR al inicio
- **Sigmoide**: variante con transición más suave en los extremos

Analizamos las curvas de SNR($t$) = $\bar\alpha_t / (1 - \bar\alpha_t)$ y el timestep donde SNR = 1 (punto de igual señal y ruido), que varía considerablemente entre schedules.

#### Extra: v-prediction

Implementamos la parametrización $v$ de Salimans & Ho (2022), que predice $v_t = \sqrt{\bar\alpha_t}\,\varepsilon - \sqrt{1-\bar\alpha_t}\,x_0$ en lugar del ruido directo, y realizamos la ablación de las tres parametrizaciones (ε, $x_0$, $v$) usando la misma arquitectura.

---

## Estructura del proyecto

```
.
├── ddpm/
│   ├── diffusion.py        # GaussianDiffusion: q_sample, L_simple, muestreo ancestral
│   ├── unet.py             # U-Net: ResBlocks, GroupNorm, embedding sinusoidal, attention
│   ├── ema.py              # Exponential Moving Average (decay=0.9999)
│   └── ddim.py             # DDIM sampler sobre pesos DDPM
│
├── extras/
│   ├── latent_interpolation.py   # Interpolación en espacio ruidoso (Sec. 4.4)
│   ├── ablation_schedules.py     # Comparativa lineal / coseno / sigmoide
│   └── v_prediction.py           # ε-pred vs x₀-pred vs v-pred unificados
│
├── demo/
│   └── app.py              # Demo interactiva con Gradio
│
├── data/
│   └── datasets.py         # DataLoaders: MNIST, Fashion-MNIST, CIFAR-10
│
├── scripts/
│   ├── plot_style.py                  # Tema visual unificado (fondo oscuro)
│   ├── viz_diffusion.py               # Visualizaciones del proceso de difusión
│   ├── viz_training.py                # Curvas de loss y gradientes
│   ├── viz_metrics.py                 # FID, IS, comparativas de muestreo
│   ├── viz_ablation.py                # Plots de ablaciones
│   ├── viz_interpolation.py           # Grids de interpolación
│   ├── plot_from_logs.py              # Genera todas las gráficas desde el JSONL
│   ├── recover_metrics_from_tb.py     # Recupera métricas de TensorBoard a JSONL
│   └── generate_all_showcase_plots.py # Pipeline completo de visualizaciones
│
├── utils/
│   ├── seeding.py          # Reproducibilidad: torch, numpy, random, cuDNN
│   ├── checkpointing.py    # Guarda/carga modelo + EMA + optimizer + RNG state
│   ├── diagnostics.py      # Normas L2 de gradientes por capa
│   └── metrics_logger.py   # Logger JSONL independiente de TensorBoard
│
├── configs/
│   ├── cifar10.yaml        # ch=128, ch_mult=(1,2,2,2), T=1000, lr=2e-4
│   └── mnist.yaml          # Configuración reducida para verificación rápida
│
├── notebooks/
│   └── showcase_lucirme.ipynb   # Análisis completo con visualizaciones
│
├── train.py                # Entrenamiento: AMP, EMA, checkpointing, TensorBoard
├── eval.py                 # Evaluación: FID (50k muestras, pesos EMA), IS, NLL
└── requirements.txt
```

---

## Instalación y uso

```bash
# Clonar e instalar dependencias
git clone https://github.com/AlegreVentura/denoising-diffusion-pytorch
cd denoising-diffusion-pytorch
pip install -r requirements.txt
```

### Entrenamiento

```bash
# Verificar correctitud primero (~3h en RTX 3060 Ti)
python train.py --config configs/mnist.yaml

# CIFAR-10 completo (~11 días a 800k pasos, ~3 días a 200k con FID útil)
python train.py --config configs/cifar10.yaml

# Reanudar desde checkpoint
python train.py --config configs/cifar10.yaml --resume checkpoints/cifar10/latest.pt
```

### Evaluación (protocolo correcto)

```bash
# FID con 50k muestras y pesos EMA — no reducir a 10k ni usar pesos raw
python eval.py --config configs/cifar10.yaml --checkpoint checkpoints/cifar10/best.pt

# Con DDIM (mucho más rápido, FID comparable)
python eval.py --config configs/cifar10.yaml --checkpoint checkpoints/cifar10/best.pt \
               --ddim --ddim_steps 100
```

### Generar gráficas

```bash
# Recuperar métricas del run actual desde TensorBoard
python scripts/recover_metrics_from_tb.py --checkpoint_dir checkpoints/cifar10

# Generar todas las gráficas de entrenamiento
python scripts/plot_from_logs.py --checkpoint_dir checkpoints/cifar10 --show

# Pipeline completo: muestras, cadena inversa, DDIM, interpolación
python scripts/generate_all_showcase_plots.py --show
```

### Demo interactiva

```bash
python demo/app.py --checkpoint checkpoints/cifar10/best.pt --config configs/cifar10.yaml
```

---

## Resultados

### MNIST (100k pasos, ~3h en RTX 3060 Ti)

El modelo converge en los primeros 5k pasos. Con 100k pasos el val loss se estabiliza en ~0.021, train y val prácticamente solapados (sin overfitting). Las normas de gradiente se mantienen estables en ~0.09 durante todo el entrenamiento, siempre por debajo del umbral de clip de 1.0 — señal de que el flujo de gradientes es sano.

### CIFAR-10 (entrenamiento en curso)

Con 38k pasos (~5% del run completo de 800k) la loss ya está en ~0.027, lo que indica convergencia rápida al inicio. Se espera un FID entre 25–40 a 200k pasos en una GPU de consumidor. El FID de 3.17 del paper requiere 800k pasos en una TPU v3-8.

> **Nota sobre reproducibilidad**: el FID depende críticamente del protocolo de evaluación. Usar pesos sin EMA o menos de 50k muestras infla el FID artificialmente (~12–13 en lugar de ~3.1 con el checkpoint oficial). Documentamos y seguimos el protocolo correcto en `eval.py`.

### Sobre las métricas de clasificación en la rúbrica

Las métricas Precision/Recall/F1/Accuracy de clasificación no aplican a un modelo generativo incondicional como DDPM: no existe una etiqueta "correcta" por imagen generada. En su lugar usamos métricas generativas estándar:

- **FID** (Frechet Inception Distance): distancia entre distribuciones en el espacio de features de InceptionV3
- **IS** (Inception Score): nitidez y diversidad conjuntas
- **NLL en bits/dim**: verosimilitud del modelo via el VLB
- **Precision/Recall generativos** (Kynkaanniemi et al., 2019): precision ≈ fidelidad de las muestras, recall ≈ cobertura del manifold real

---

## Referencias

```bibtex
@inproceedings{ho2020ddpm,
  title   = {Denoising Diffusion Probabilistic Models},
  author  = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle = {NeurIPS},
  year    = {2020},
  url     = {https://arxiv.org/abs/2006.11239}
}

@article{song2021ddim,
  title   = {Denoising Diffusion Implicit Models},
  author  = {Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal = {ICLR},
  year    = {2021},
  url     = {https://arxiv.org/abs/2010.02502}
}

@article{nichol2021improved,
  title   = {Improved Denoising Diffusion Probabilistic Models},
  author  = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
  journal = {ICML},
  year    = {2021},
  url     = {https://arxiv.org/abs/2102.09672}
}

@inproceedings{kynkaanniemi2019improved,
  title   = {Improved Precision and Recall Metric for Assessing Generative Models},
  author  = {Kynk{\"a}{\"a}nniemi, Tuomas and Karras, Tero and Laine, Samuli and Lehtinen, Jaakko and Aila, Timo},
  booktitle = {NeurIPS},
  year    = {2019}
}
```

---

<div align="center">
  <sub>
    Proyecto Final — Aprendizaje Profundo · Junio 2026<br/>
    Implementación propia basada en el paper original y el repositorio oficial
    <a href="https://github.com/hojonathanho/diffusion">hojonathanho/diffusion</a>
  </sub>
</div>
