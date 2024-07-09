"""
Not in original Open-MAGVIT2.
"""

import json
from dataclasses import dataclass


@dataclass
class VQConfig:
    # Model Arch
    in_channels: int = 3
    z_channels: int = 18
    out_channels: int = 3
    base_channels: int = 128  # Initial hidden width, all subsequent blocks are multiples (`ch_mult`) of this width
    ch_mult: tuple[int] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 2

    # Loss Config (uncertain about some of the types)
    # Hardcoding for `magvit2.modules.losses.vqperceptual.VQLPIPSWithDiscriminator`
    disc_conditional: bool = False
    disc_in_channels: int = 3
    disc_start: int = 0  # from 0 epoch
    disc_loss: str = "hinge"
    disc_ndf: int = 64
    disc_num_layers: int = 3
    use_actnorm: bool = False
    disc_factor: float = 1.0
    disc_weight: float = 0.8
    gen_loss_weight: float = 0.1
    lecam_loss_weight: float = 0.005
    codebook_weight: float = 0.1
    commit_weight: float = 0.25
    pixelloss_weight: float = 1.0
    perceptual_weight: float = 1.0
    codebook_enlarge_ratio: float = 0
    codebook_enlarge_steps: int = 2000

    num_codebooks: int = 1
    codebook_size: int = 262144
    sample_minimization_weight: float = 1.0
    batch_maximization_weight: float = 1.0
    token_factorization: bool = False

    # TODO: duplicated from GenieConfig
    def save_pretrained(self, json_path):
        with open(json_path, "w") as f:
            json.dump(vars(self), f)

    @classmethod
    def from_pretrained(cls, json_path):
        with open(json_path, "r") as f:
            config = json.load(f)

        return cls(**config)
