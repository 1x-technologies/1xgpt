from typing import Callable

import torch
import torchvision.transforms.functional as transforms_f
from einops import rearrange

from genie.factorization_utils import factorize_labels


class AvgMetric:
    """ Records a running sum and count to compute the mean. """
    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, val, batch_size=1):
        self.total += val * batch_size
        self.count += batch_size

    def update_list(self, flat_vals):
        self.total += sum(flat_vals)
        self.count += len(flat_vals)

    def mean(self):
        return self.total / self.count


def decode_tokens(reshaped_token_ids: torch.LongTensor, decode_latents: Callable) -> torch.ByteTensor:
    """
    Converts quantized latent space tokens to images.

    Args:
        reshaped_token_ids: shape (B, T, H, W).
        decode_latents: instance of `decode_latents_wrapper()`

    Returns:
        (B, T, 3, 256, 256)
    """
    decoded_imgs = decode_latents(rearrange(reshaped_token_ids, "b t h w -> (b t) h w").cpu().numpy())
    decoded_tensor = torch.stack([transforms_f.pil_to_tensor(pred_img) for pred_img in decoded_imgs])
    return rearrange(decoded_tensor, "(b t) c H W -> b t c H W", b=reshaped_token_ids.size(0))


def compute_loss(
        labels_flat: torch.LongTensor,
        factored_logits: torch.FloatTensor,
        num_factored_vocabs: int = 2,
        factored_vocab_size: int = 512,
) -> float:
    """
    If applicable (model returns logits), compute the cross entropy loss.
    In the case of a factorized vocabulary, sums the cross entropy losses for each vocabulary.

    Assuming all submissions use the parametrization of num_factored_vocabs = 2, factored_vocab_size = 512

    Args:
        labels_flat: size (B, T*H*W) corresponding to flattened, tokenized images.
        factored_logits: size (B, factored_vocab_size, num_factored_vocabs, T-1, H, W).
            E.g. output of `genie.evaluate.GenieEvaluator.predict_zframe_logits()`
        num_factored_vocabs: Should be 2 for v1.0 of the challenge.
        factored_vocab_size: Should be 512 for v1.0 of the challenge.
    Returns:
        Cross entropy loss
    """
    assert factored_logits.dim() == 6 \
           and factored_logits.size()[:3] == (labels_flat.size(0), factored_vocab_size, num_factored_vocabs), \
           f"Shape of `logits` should be (B, {factored_vocab_size}, {num_factored_vocabs}, T-1, H, W)"
    t = factored_logits.size(3) + 1
    h, w = factored_logits.size()[-2:]
    assert t * h * w == labels_flat.size(1), "Shape of `factored_logits` does not match flattened latent image size."

    labels_THW = rearrange(labels_flat, "b (t h w) -> b t h w", t=t, h=h, w=w)
    labels_THW = labels_THW[:, 1:].to(factored_logits.device)

    factored_labels = factorize_labels(labels_THW, num_factored_vocabs, factored_vocab_size)
    return torch.nn.functional.cross_entropy(factored_logits, factored_labels, reduction="none")\
        .sum(dim=1).mean().item()  # Final loss is the sum of the two losses across the size-512 vocabularies


def compute_lpips(frames_a: torch.ByteTensor, frames_b: torch.ByteTensor, lpips_func: Callable) -> list:
    """
    Given two batches of video data, of shape (B, T, 3, 256, 256), computes the LPIPS score on frame-by-frame level.
    Cannot use `lpips_func` directly because it expects at most 4D input.
    """
    # LPIPS expects pixel values between [-1, 1]
    flattened_a, flattened_b = [rearrange(frames / 127.5 - 1, "b t c H W -> (b t) c H W")
                                for frames in (frames_a, frames_b)]
    return lpips_func(flattened_a, flattened_b).flatten().tolist()
