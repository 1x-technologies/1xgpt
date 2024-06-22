from typing import Callable

import torch
import torchvision.transforms.functional as transforms_f
from einops import rearrange


class AvgMetric:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size=1):
        self.sum += val * batch_size
        self.count += batch_size

    def update_list(self, flat_vals):
        self.sum += sum(flat_vals)
        self.count += len(flat_vals)

    def mean(self):
        return self.sum / self.count


def decode_tokens(reshaped_token_ids: torch.LongTensor, decode_latents: Callable) -> torch.ByteTensor:
    """
    Converts quantized latent space tokens to images.

    Args:
        reshaped_token_ids: shape (B, T, H, W).
        decode_latents: instance of `decode_latents_wrapper()`

    Returns:
        (B, T, 3, 160, 160)
    """
    decoded_imgs = decode_latents(rearrange(reshaped_token_ids, "b t h w -> (b t) h w").numpy())
    decoded_tensor = torch.stack([transforms_f.pil_to_tensor(pred_img) for pred_img in decoded_imgs])
    return rearrange(decoded_tensor, "(b t) c H W -> b t c H W", b=reshaped_token_ids.size(0))


def compute_loss_and_acc(input_ids: torch.LongTensor, logits: torch.FloatTensor) -> tuple[float, float]:
    """
    If applicable (Evaluator can return logits), compute the cross entropy loss and predicted token accuracy.

    Args:
        input_ids: LongTensor of size (B, T*H*W) corresponding to flattened, tokenized images.
        logits: FloatTensor of size (B, C, T-1, H, W). E.g. output of `LlamaEvaluator.predict_zframe_logits()`

    Returns:
        Cross entropy loss and predicted token accuracy.
    """

    assert logits.dim() == 5 and input_ids.size(0) == logits.size(0) and logits.size(1) == 1000, \
        "Shape of `logits` should be (B, C, T-1, h, w)"
    t = logits.size(2) + 1
    h, w = logits.size()[-2:]
    assert t * h * w == input_ids.size(1), "Shape of `logits` does not match flattened latent image size."
    input_ids = rearrange(input_ids, "b (t h w) -> b t h w", t=t, h=h, w=w)
    labels = input_ids[:, 1:].to(logits.device)
    top_preds = torch.argmax(logits, dim=1)
    return torch.nn.functional.cross_entropy(logits, labels).item(), (labels == top_preds).float().mean().item()


def compute_lpips(frames_a: torch.ByteTensor, frames_b: torch.ByteTensor, lpips_func: Callable) -> list:
    """
    Given two batches of video data, of shape (B, T, 3, 160, 160), computes the LPIPS score on frame-by-frame level.
    Cannot use `lpips_func` directly because it expects at most 4D input.
    """
    flattened_a, flattened_b = [rearrange(frames, "b t c H W -> (b t) c H W")
                                for frames in (frames_a, frames_b)]
    return lpips_func(flattened_a, flattened_b).flatten().tolist()
