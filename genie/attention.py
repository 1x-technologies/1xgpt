import os
import warnings
from typing import Optional

import torch
from torch import nn

from xformers.ops import LowerTriangularMask, memory_efficient_attention, unbind


class SelfAttention(nn.Module):
    # NOTE: Mem-eff attention from xformers is actually Flash Attention 2
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        qkv_norm: bool = False,
        use_mup: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 8/head_dim if use_mup else head_dim**-0.5  # Scaling by 8 to be equal when head_dim=64
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.qkv_norm = qkv_norm
        if qkv_norm:
            # qk normalization https://arxiv.org/pdf/2302.05442
            # Note that LN is done in fp32, so they have to be
            self.norm = nn.LayerNorm(head_dim, eps=1e-05)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        if self.qkv_norm:
            q = self.norm(q)
            k = self.norm(k)
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)

        attn_bias = LowerTriangularMask() if causal else None

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        return x
