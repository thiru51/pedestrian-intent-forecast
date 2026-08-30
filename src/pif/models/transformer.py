"""A small temporal transformer, written out rather than pulled from
torch.nn.TransformerEncoder.

Two reasons it is hand-written. First, nn.TransformerEncoder carries a lot of
branching for masks and nested tensors that ONNX export has historically been
awkward about, and this repo has to export cleanly. Second, at T=16 the
attention is 16x16 and the interesting engineering is entirely in the
projection widths and the pre-norm placement, which are easier to reason about
when they are in front of you.

Pre-norm (norm inside the residual branch) rather than post-norm: at 2-4 layers
either trains, but pre-norm does not need a warmup schedule to stay stable,
and one less moving part is worth it.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        # No causal mask: the window is a completed observation, so every frame
        # in it may look at every other. The forecast is about frames outside
        # the window, which none of these attend to by construction.
        weights = self.attn_drop(scores.softmax(dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj_drop(self.proj(out))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TemporalTransformer(nn.Module):
    """(B, T, in_dim) -> (B, T, d_model) tokens plus a (B, d_model) summary.

    Positional information is a learned per-step embedding, not sinusoids: T is
    fixed at 16 and the steps are 33 ms apart, so there is no length
    generalisation to preserve and a learned table fits the fixed geometry
    better.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        window: int = 16,
    ):
        super().__init__()
        self.window = window
        self.d_model = d_model
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        # Learned pooling over time: which frames of the window matter is
        # itself a thing worth learning, and it stays a plain matmul for ONNX.
        self.pool_score = nn.Linear(d_model, 1)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x) + self.pos[:, : x.shape[1]]
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        weights = self.pool_score(h).softmax(dim=1)  # (B, T, 1)
        summary = (h * weights).sum(dim=1)
        return h, summary


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
