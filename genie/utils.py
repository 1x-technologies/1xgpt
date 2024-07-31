import torch

from genie.config import GenieConfig
from genie.st_mask_git import STMaskGIT


def convert_lightning_checkpoint(lightning_checkpoint, num_layers, num_heads, d_model, save_dir):
    """
    v0.0.1 saved models in Lightning checkpoints, this can convert Lightning checkpoints to HF checkpoints.
    """
    config = GenieConfig(num_layers=num_layers, num_heads=num_heads, d_model=d_model)
    model = STMaskGIT(config)

    lightning_checkpoint = torch.load(lightning_checkpoint)
    model_state_dict = lightning_checkpoint["state_dict"]

    # Remove `model.` prefix
    model_state_dict = {name.replace("model.", ""): params for name, params in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    model.save_pretrained(save_dir)
