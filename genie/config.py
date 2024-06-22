import json
from dataclasses import dataclass


@dataclass
class GenieConfig:
    num_layers: int
    num_heads: int
    d_model: int
    T: int = 16  # temporal sequence length
    S: int = 400  # spatial sequence length, e.g. 400 for 20x20
    image_vocab_size: int = 1000  # image_vocab_size:  number of distinct image tokens;
    # actual model vocab size is bigger to include special (e.g. mask) tokens.
    use_mup: bool = False

    # MaskGIT training
    min_mask_rate: float = 0.5
    max_mask_rate: float = 1.0
    max_random_token_rate: float = 0.1

    # Attention
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    qk_norm: bool = True

    # MLP
    mlp_ratio: float = 4.0
    mlp_drop: float = 0.0
    mlp_bias: bool = True

    def save_pretrained(self, json_path):
        with open(json_path, "w") as f:
            json.dump(vars(self), f)

    @classmethod
    def from_pretrained(cls, json_path):
        with open(json_path, "r") as f:
            config = json.load(f)

        return cls(**config)

    def shallow_copy(self):
        return GenieConfig(**vars(self))
