from torch import nn, Tensor
from typing import Optional
from einops import rearrange

from baselines.attention import SelfAttention

# Parameters used by 10B Genie model:
# Encoder
# num_layers=12
# d_model=512
# num_heads=8
# k/q_size = 64

# Decoder
# num_layers=20
# d_model=1024
# num_heads=16
# k/q_size=64

class Mlp(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, drop: float = 0.0, bias: bool = True
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class STBlock(nn.Module):
    # See Figure 4 of https://arxiv.org/pdf/2402.15391.pdf
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        qkv_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.Identity() if qkv_norm else nn.LayerNorm(dim, eps=1e-05)
        # sequence dim is over each frame's 32x32 patch tokens
        self.spatial_attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, qkv_norm=qkv_norm)
        # sequence dim is over time sequence (16)
        self.temporal_attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, qkv_norm=qkv_norm)
        
        self.norm2 = nn.Identity() if qkv_norm else nn.LayerNorm(dim, eps=1e-05)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop, bias=ffn_bias)
        
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



class STTransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([STBlock(**kwargs) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class STTransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([STBlock(**kwargs) for _ in range(num_layers)])

    def forward(self, tgt):
        x = tgt
        for layer in self.layers:
            x = layer(x)
        return x
