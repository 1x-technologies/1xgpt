import torch
from torch import nn

from xformers.ops import LowerTriangularMask, memory_efficient_attention, unbind


class SelfAttention(nn.Module):
    # NOTE: Mem-eff attention from xformers is actually Flash Attention 2
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        num_heads, d_model = config.num_heads, config.d_model
        self.head_dim = d_model // num_heads

        # Scaling by 8 to be equal when head_dim=64
        self.scale = 8/self.head_dim if self.config.use_mup else self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(self.config.attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=config.proj_bias)
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            # qk normalization https://arxiv.org/pdf/2302.05442
            # Note that LN is done in fp32, so they have to be
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.config.num_heads, self.head_dim)
        q, k, v = unbind(qkv, 2)
        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)

        attn_bias = LowerTriangularMask() if causal else None
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale)

        x = x.reshape([B, N, C])

        x = self.proj(x)
        return x
