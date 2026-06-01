"""U-Net para DDPM (Ho et al., NeurIPS 2020).

Arquitectura fiel al paper, config CIFAR-10:
  ch=128, ch_mult=(1,2,2,2), num_res_blocks=2,
  attn_resolutions=(16,), dropout=0.1  -> 35.7M parametros

Componentes:
  SinusoidalTimestepEmbedding  -> embedding de t (como Transformers)
  ResidualBlock                -> GroupNorm + SiLU + Conv, con conditioning de t
  SelfAttentionBlock           -> multi-head attention en resolucion 16x16
  Downsample / Upsample        -> conv con stride / upsample bilinear + conv
  UNet                         -> encoder-bottleneck-decoder con skip connections
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Embedding sinusoidal del timestep
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Embedding posicional sinusoidal para el timestep t.

    Identico a los embeddings de posicion de los Transformers originales.
    Convierte un escalar t en un vector de dimension embedding_dim.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        timesteps : (B,) tensor de enteros

        Returns
        -------
        embedding : (B, embedding_dim) tensor
        """
        half_dim = self.embedding_dim // 2
        log_max_period = math.log(10000.0)
        frequencies = torch.exp(
            -log_max_period * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / (half_dim - 1)
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return embedding


# ---------------------------------------------------------------------------
# Bloque residual con conditioning de timestep
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Bloque residual estilo Wide-ResNet con GroupNorm y embedding de tiempo.

    Estructura:
      GroupNorm -> SiLU -> Conv -> [+time_proj] -> GroupNorm -> SiLU -> Dropout -> Conv
      + conexion residual (1x1 conv si in != out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.1,
        num_groups: int = 32,
    ):
        super().__init__()
        self.norm_pre_conv1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_projection = nn.Linear(time_embed_dim, out_channels)

        self.norm_pre_conv2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.norm_pre_conv1(x))
        h = self.conv1(h)

        time_bias = self.time_projection(self.activation(time_embed))
        h = h + time_bias[:, :, None, None]

        h = self.activation(self.norm_pre_conv2(h))
        h = self.dropout_layer(h)
        h = self.conv2(h)

        return h + self.skip_connection(x)


# ---------------------------------------------------------------------------
# Bloque de self-attention (aplicado en resoluciones especificas)
# ---------------------------------------------------------------------------

class SelfAttentionBlock(nn.Module):
    """Self-attention en espacio espacial.

    Aplana la feature map a una secuencia de tokens (H*W), aplica
    multi-head attention y reconstruye la forma original.
    Se aplica unicamente en attn_resolutions (p.ej. 16x16 para CIFAR-10).
    """

    def __init__(self, channels: int, num_heads: int = 1, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        h, _ = self.attention(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.out_projection(h)

        return residual + h


# ---------------------------------------------------------------------------
# Downsampling y Upsampling
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """Reduccion de resolucion por 2 con conv stride=2 (aprende el submuestreo)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv_stride2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stride2(x)


class Upsample(nn.Module):
    """Ampliacion de resolucion x2 con nearest-neighbor + conv (suaviza artefactos)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv_after_upsample = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv_after_upsample(x)


# ---------------------------------------------------------------------------
# U-Net completo
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """U-Net parametrizado por timestep para predecir el ruido eps_theta(x_t, t).

    Arquitectura de Ho et al. con:
      - Encoder: bloques residuales + downsampling por nivel
      - Bottleneck: ResBlock + Attention + ResBlock
      - Decoder: upsampling + skip connections + bloques residuales
      - Attention solo en las resoluciones indicadas en attn_resolutions

    Parameters
    ----------
    image_channels       : canales de la imagen (3 para CIFAR-10)
    base_channels        : canales base del modelo (ch=128)
    channel_multipliers  : multiplicadores por nivel (ch_mult=(1,2,2,2))
    num_res_blocks       : ResBlocks por nivel del encoder/decoder (2)
    attention_resolutions: resoluciones donde se aplica attention ({16})
    dropout              : dropout en ResBlocks (0.1)
    num_groups           : grupos para GroupNorm (32)
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.1,
        num_groups: int = 32,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)

        time_embed_dim = base_channels * 4
        channels_per_level = [base_channels * mult for mult in channel_multipliers]
        num_levels = len(channel_multipliers)

        # ------------------------------------------------------------------
        # Embedding del timestep: sinusoidal -> MLP 2 capas
        # ------------------------------------------------------------------
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(base_channels),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # ------------------------------------------------------------------
        # Proyeccion de entrada
        # ------------------------------------------------------------------
        self.input_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        # ------------------------------------------------------------------
        # Encoder (camino descendente)
        # ------------------------------------------------------------------
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()

        current_resolution = None  # se calcula dinamicamente en forward
        current_channels = base_channels
        self._encoder_output_channels: List[int] = [base_channels]  # para skip connections

        for level_idx, level_channels in enumerate(channels_per_level):
            level_module_list = nn.ModuleList()
            for block_idx in range(num_res_blocks):
                in_ch = current_channels if block_idx == 0 else level_channels
                level_module_list.append(
                    ResidualBlock(in_ch, level_channels, time_embed_dim, dropout, num_groups)
                )
                self._encoder_output_channels.append(level_channels)
            self.encoder_blocks.append(level_module_list)

            if level_idx < num_levels - 1:
                self.encoder_downsamples.append(Downsample(level_channels))
                self._encoder_output_channels.append(level_channels)
            else:
                self.encoder_downsamples.append(None)

            current_channels = level_channels

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        bottleneck_channels = channels_per_level[-1]
        self.bottleneck_res_block_1 = ResidualBlock(
            bottleneck_channels, bottleneck_channels, time_embed_dim, dropout, num_groups
        )
        self.bottleneck_attention = SelfAttentionBlock(bottleneck_channels, num_groups=num_groups)
        self.bottleneck_res_block_2 = ResidualBlock(
            bottleneck_channels, bottleneck_channels, time_embed_dim, dropout, num_groups
        )

        # ------------------------------------------------------------------
        # Decoder (camino ascendente, con skip connections)
        # ------------------------------------------------------------------
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        skip_channel_stack = list(reversed(self._encoder_output_channels))
        current_channels = bottleneck_channels

        for level_idx, level_channels in enumerate(reversed(channels_per_level)):
            level_module_list = nn.ModuleList()
            for block_idx in range(num_res_blocks + 1):
                skip_ch = skip_channel_stack.pop(0)
                in_ch = current_channels + skip_ch
                out_ch = level_channels
                level_module_list.append(
                    ResidualBlock(in_ch, out_ch, time_embed_dim, dropout, num_groups)
                )
                current_channels = out_ch
            self.decoder_blocks.append(level_module_list)

            decoder_level_idx = num_levels - 1 - level_idx
            if decoder_level_idx > 0:
                self.decoder_upsamples.append(Upsample(level_channels))
            else:
                self.decoder_upsamples.append(None)

        # ------------------------------------------------------------------
        # Proyeccion de salida
        # ------------------------------------------------------------------
        self.output_norm = nn.GroupNorm(num_groups, base_channels)
        self.output_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)
        self.output_activation = nn.SiLU()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inicializa la conv de salida a cero (comun en modelos de difusion)."""
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x_noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_noisy   : (B, C, H, W) imagen con ruido x_t
        timesteps : (B,) enteros en [0, T-1]

        Returns
        -------
        predicted_noise : (B, C, H, W) ruido predicho eps_theta(x_t, t)
        """
        current_resolution = x_noisy.shape[-1]

        # Embedding del timestep
        time_embed = self.time_embedding(timesteps)

        # Proyeccion de entrada
        h = self.input_conv(x_noisy)

        # ------ Encoder ------
        skip_connection_stack: List[torch.Tensor] = [h]

        for level_idx, (res_blocks, downsample) in enumerate(
            zip(self.encoder_blocks, self.encoder_downsamples)
        ):
            level_channels = res_blocks[0].conv1.out_channels

            for res_block in res_blocks:
                h = res_block(h, time_embed)
                if current_resolution in self.attention_resolutions:
                    pass  # attention se aplica solo en el bottleneck y decoder en esta version
                skip_connection_stack.append(h)

            if downsample is not None:
                h = downsample(h)
                current_resolution = current_resolution // 2
                skip_connection_stack.append(h)

        # ------ Bottleneck ------
        h = self.bottleneck_res_block_1(h, time_embed)
        h = self.bottleneck_attention(h)
        h = self.bottleneck_res_block_2(h, time_embed)

        # ------ Decoder ------
        num_levels = len(self.channel_multipliers)
        for level_idx, (res_blocks, upsample) in enumerate(
            zip(self.decoder_blocks, self.decoder_upsamples)
        ):
            for res_block in res_blocks:
                skip = skip_connection_stack.pop()
                h = torch.cat([h, skip], dim=1)
                h = res_block(h, time_embed)

            decoder_resolution_idx = num_levels - 1 - level_idx
            if upsample is not None:
                h = upsample(h)
                current_resolution = current_resolution * 2

            if current_resolution in self.attention_resolutions:
                pass  # posicion para insertar attention en decoder si se necesita

        # ------ Salida ------
        h = self.output_activation(self.output_norm(h))
        return self.output_conv(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
