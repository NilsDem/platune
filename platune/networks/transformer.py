import torch
from einops.layers.torch import Rearrange
from torch import nn
from einops import rearrange
import gin
import numpy as np


class PositionalEmbedding(nn.Module):

    def __init__(
        self,
        num_channels: int,
        max_positions: int,
        factor: float,
        endpoint: bool = False,
        rearrange: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.factor = factor
        self.rearrange = (Rearrange("b (f c) -> b (c f)", f=2)
                          if rearrange else nn.Identity())

    def forward(self, x: torch.Tensor):
        x = x.view(-1)
        x = x * self.factor
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            device=x.device,
        ).float()
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions)**freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return self.rearrange(x)


class MHAttention(nn.Module):

    def __init__(self, is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):
        q, k, v = [
            rearrange(x, "bs n (h d) -> bs h n d", h=self.n_heads)
            for x in [q, k, v]
        ]
        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
            dropout_p=self.dropout_level if self.training else 0,
        )
        out = rearrange(out, "bs h n d -> bs n (h d)", h=self.n_heads)
        return out


@gin.configurable
class SelfAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 is_causal=False,
                 dropout_level=0.0,
                 n_heads=8):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v)


class MLPSepConv(nn.Module):

    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        """see: https://github.com/ofsoundof/LocalViT"""
        super().__init__()
        self.mlp = nn.Sequential(
            # this Conv with kernel size 1 is equivalent to the Linear layer in a "regular" transformer MLP
            nn.Conv1d(embed_dim,
                      mlp_multiplier * embed_dim,
                      kernel_size=1,
                      padding="same"),
            nn.Conv1d(
                mlp_multiplier * embed_dim,
                mlp_multiplier * embed_dim,
                kernel_size=3,
                padding="same",
                groups=mlp_multiplier * embed_dim,
            ),  # <- depthwise conv
            nn.GELU(),
            nn.Conv1d(mlp_multiplier * embed_dim,
                      embed_dim,
                      kernel_size=1,
                      padding="same"),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        x = rearrange(x, "b t c -> b c t")
        x = self.mlp(x)
        x = rearrange(x, "b c t -> b t c")
        return x


class DecoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        cond_dim: int,
        is_causal: bool,
        mlp_multiplier: int,
        dropout_level: float,
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.self_attention = SelfAttention(embed_dim,
                                            is_causal,
                                            dropout_level,
                                            n_heads=embed_dim // 64)
        
        self.mlp = MLPSepConv(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)  # , elementwise_affine=False) ?
        self.norm3 = nn.LayerNorm(embed_dim)

        if self.cond_dim > 0:
            self.linear = nn.Linear(cond_dim, 2 * embed_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.self_attention(self.norm1(x)) + x
        # AdaLN
        x = self.norm2(x)
        if self.cond_dim > 0:
            assert cond is not None
            alpha, beta = self.linear(cond).chunk(2, dim=-1)
            x = x * (1 + alpha.unsqueeze(1)) + beta.unsqueeze(1)
        
        x = self.mlp(self.norm3(x)) + x
        return x


class DenoiserTransBlock(nn.Module):

    def __init__(
        self,
        n_channels: int = 64,
        seq_len: int = 32,
        mlp_multiplier: int = 4,
        embed_dim: int = 256,
        cond_dim: int = 128,
        dropout: float = 0.1,
        n_layers: int = 4,
        is_causal: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier
        self.patchify_and_embed = nn.Sequential(
            Rearrange("b c t -> b t c"),
            nn.Linear(n_channels, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.rearrange2 = Rearrange("b t c -> b c t", )
        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer("precomputed_pos_enc", torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim=self.embed_dim,
                cond_dim=cond_dim,
                is_causal=is_causal,
                mlp_multiplier=self.mlp_multiplier,
                dropout_level=self.dropout,
            ) for _ in range(self.n_layers)
        ])

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, n_channels), self.rearrange2)

    def forward(self, x, features):

        #x = torch.cat([x, noise_level], dim=1)

        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
        x = x + self.pos_embed(pos_enc)

        for block in self.decoder_blocks:
            x = block(x, cond=features)

        return self.out_proj(x)


@gin.configurable
class Denoiser(nn.Module):

    def __init__(
        self,
        n_channels: int,
        seq_len: int = 32,
        embed_dim: int = 256,
        noise_embed_dims: int = 128,
        n_layers: int = 6,
        mlp_multiplier: int = 2,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels

        self.fourier_feats = PositionalEmbedding(
            num_channels=noise_embed_dims,
            max_positions=10_000,
            factor=100.0
        )

        self.embedding = nn.Sequential(
            nn.Linear(noise_embed_dims, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.denoiser_trans_block = DenoiserTransBlock(
            n_channels=n_channels,
            seq_len=seq_len,
            mlp_multiplier=mlp_multiplier,
            embed_dim=embed_dim,
            cond_dim=embed_dim,
            dropout=dropout,
            n_layers=n_layers,
            is_causal=causal)

    @property
    def name(self):
        return "transformer"

    def forward(self, x, time):
        time = time.reshape(-1)
        noise_level = self.fourier_feats(time)

        features = self.embedding(noise_level)

        x = self.denoiser_trans_block(x, features=features)

        return x
