from torch import nn, Tensor
from typing import Optional
from einops import rearrange

from genie.attention import SelfAttention
from genie.config import GenieConfig


class Mlp(nn.Module):
    def __init__(self, config: GenieConfig) -> None:
        super().__init__()
        hidden_dim = int(config.d_model * config.mlp_ratio)
        self.fc1 = nn.Linear(config.d_model, hidden_dim, bias=config.mlp_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, config.d_model, bias=config.mlp_bias)
        self.drop = nn.Dropout(config.mlp_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class STBlock(nn.Module):
    # See Figure 4 of https://arxiv.org/pdf/2402.15391.pdf
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.Identity() if config.qk_norm else nn.LayerNorm(config.d_model, eps=1e-05)
        # sequence dim is over each frame's 20x20 patch tokens
        self.spatial_attn = SelfAttention(config)
        # sequence dim is over time sequence (16)
        self.temporal_attn = SelfAttention(config)
        
        self.norm2 = nn.Identity() if config.qk_norm else nn.LayerNorm(config.d_model, eps=1e-05)
        self.mlp = Mlp(config)
        
    def forward(self, x_TSC: Tensor) -> Tensor:
        # Process attention spatially
        T, S = x_TSC.size(1), x_TSC.size(2)
        x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))
        # Process attention temporally
        x_TC = rearrange(x_SC, '(B T) S C -> (B S) T C', T=T)
        x_TC = x_TC + self.temporal_attn(x_TC, causal=True)
        # Apply the MLP
        x_TC = x_TC + self.mlp(self.norm2(x_TC))
        x_TSC = rearrange(x_TC, '(B S) T C -> B T S C', S=S)
        return x_TSC


class STTransformerDecoder(nn.Module):
    def __init__(self, config: GenieConfig):
        super().__init__()
        self.layers = nn.ModuleList([STBlock(config) for _ in range(config.num_layers)])

    def forward(self, tgt):
        x = tgt
        for layer in self.layers:
            x = layer(x)
        return x
