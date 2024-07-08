#!/usr/bin/env python3

"""Script to decode tokenized video into images/video."""

import argparse
import math
import os
from PIL import Image

import numpy as np
import torch
import torch.distributed.optim
import torch.utils.checkpoint
import torch.utils.data
import torchvision.transforms.v2.functional as transforms_f
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from data import RawTokenDataset
# # Custom mod of diffusers UNet2DConditionModel
# from diffusers import AutoencoderKL, StableDiffusionInstructPix2PixPipeline
# from decoder.unet_2d_condition import UNet2DConditionModel2
from magvit2.config import VQConfig
from magvit2.models.lfqgan import VQModel


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tokenized video as GIF or comic.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame skip",
    )
    parser.add_argument(
        "--token_dir",
        type=str,
        default="data/generated",
        help="Directory of tokens, in the format of `video.bin` and `metadata.json`. "
             "Visualized gifs will be written here.",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Offset to start generating images from"
    )
    parser.add_argument(
        "--fps", type=int, default=2, help="Frames per second"
    )
    parser.add_argument(
        "--disable_comic", action="store_true",
        help="Comic generation assumes `token_dir` follows the same format as generate: e.g., "
             "`prompt | predictions | gtruth` in `video.bin`, `window_size` in `metadata.json`."
             "Therefore, comic should be disabled when visualizing videos without this format, such as the dataset."
    )
    args = parser.parse_args()

    return args


def export_to_gif(frames: list, output_gif_path: str, fps: int):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - fps (int): Desired frames per second.
    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    duration_ms = 1000 / fps
    pil_frames[0].save(output_gif_path.replace(".mp4", ".gif"),
                       format="GIF",
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=duration_ms,
                       loop=0)


def decode_latents_wrapper(batch_size=16, tokenizer_ckpt="data/magvit2.ckpt"):
    device = "cuda"
    dtype = torch.bfloat16

    model_config = VQConfig()
    model = VQModel(model_config, ckpt_path=tokenizer_ckpt)
    model = model.to(device=device, dtype=dtype)

    @torch.no_grad()
    def decode_latents(video_data):
        """
        video_data: (b, h, w), where b is different from training/eval batch size.
        """
        decoded_imgs = []

        for shard_ind in range(math.ceil(len(video_data) / batch_size)):
            batch = torch.from_numpy(video_data[shard_ind * batch_size: (shard_ind + 1) * batch_size].astype(np.int64))
            if model.use_ema:
                with model.ema_scope():
                    quant = model.quantize.get_codebook_entry(rearrange(batch, "b h w -> b (h w)"),
                                                              bhwc=batch.shape + (model.quantize.codebook_dim,)).flip(1)
                    decoded_imgs.append(((model.decode(quant.to(device=device, dtype=dtype)).detach().cpu() + 1) * 127.5).to(dtype=torch.uint8))

        return [transforms_f.to_pil_image(img) for img in torch.cat(decoded_imgs)]

    return decode_latents


@torch.no_grad()
def main():
    args = parse_args()

    # Load tokens
    token_dataset = RawTokenDataset(args.token_dir, 1, filter_interrupts=False, filter_overlaps=False)
    video_data = token_dataset.data
    metadata = token_dataset.metadata

    images = decode_latents_wrapper()(video_data[args.offset::args.stride])
    output_gif_path = os.path.join(args.token_dir, f"generated_offset{args.offset}.gif")
    export_to_gif(images, output_gif_path, args.fps)
    print(f"Saved to {output_gif_path}")

    if not args.disable_comic:
        fig, axs = plt.subplots(nrows=2, ncols=metadata["window_size"], figsize=(3 * metadata["window_size"], 3 * 2))
        for i, image in enumerate(images):
            if i < metadata["num_prompt_frames"]:
                curr_axs = [axs[0, i], axs[1, i]]
                title = "Prompt"

            elif i < metadata["window_size"]:
                curr_axs = [axs[0, i]]
                title = "Prediction"
            else:
                curr_axs = [axs[1, i - metadata["window_size"] + metadata["num_prompt_frames"]]]
                title = "Ground truth"

            for ax in curr_axs:
                ax.set_title(title)
                ax.imshow(image)
                ax.axis("off")

        output_comic_path = os.path.join(args.token_dir, f"generated_comic_offset{args.offset}.png")
        plt.savefig(output_comic_path, bbox_inches="tight")
        plt.close()
        print(f"Saved to {output_comic_path}")


if __name__ == "__main__":
    main()
