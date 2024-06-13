#!/usr/bin/env python3

"""Script to decode tokenized video into images/video."""

import argparse
import json
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.distributed.optim
import torch.utils.checkpoint
import torch.utils.data
from diffusers import AutoencoderKL, StableDiffusionInstructPix2PixPipeline
from diffusers.utils import check_min_version
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from data import RawTokenDataset
# Custom mod of diffusers UNet2DConditionModel
from decoder.unet_2d_condition import UNet2DConditionModel2

# Compilation flags
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

# Will error if the minimal version of diffusers is not installed. Remove at your own risk.
check_min_version("0.28.0")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="timbrooks/instruct-pix2pix",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_checkpoint_path",
        type=str,
        default="data/checkpoint-111000",
        help="Path to the pretrained UNet checkpoint.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=100,
        help="Number of frames to decode",
    )
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
    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=duration_ms,
                       loop=0)


def decode_latents_wrapper(pretrained_model_name_or_path="timbrooks/instruct-pix2pix",
                           unet_checkpoint_path="data/checkpoint-111000", device="cuda",
                           max_images=360):
    # VAE does image decoding
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
    ).to(device=device, dtype=torch.bfloat16)

    unet = UNet2DConditionModel2.from_pretrained(
        unet_checkpoint_path,
        subfolder="unet",
        vector_quantize=False,
    ).to(device=device, dtype=torch.bfloat16)

    # Create pipeline
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipeline.safety_checker = None  # Disabling the safety checker.
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    def decode_latents(video_data):
        """ Decompresses `video_data` from the latent space to the image space. """
        image_guidance_scale = 3.
        text_guidance_scale = 1.
        num_inference_steps = 20

        prompt_embeds = torch.zeros((1, 32, 768), device=device)
        images = []
        with torch.autocast(device, dtype=torch.bfloat16, enabled=True):
            for indices_HW in tqdm(video_data):
                input_image_tensor = pipeline.unet.fsq.implicit_codebook[indices_HW.flatten().astype(np.int32)].reshape(
                    indices_HW.shape + (-1,))

                # input_image_tensor = torch.tensor(input_image_tensor, device=device)
                input_image_tensor = rearrange(input_image_tensor, 'h w c -> 1 c h w')
                # Conv upsample
                # input_image_tensor = pipeline.unet.conv_upsample(input_image_tensor)
                # unet.vector_quantize = False
                img = pipeline(
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    image=input_image_tensor,
                    num_inference_steps=num_inference_steps,
                    image_guidance_scale=image_guidance_scale,
                    guidance_scale=text_guidance_scale,
                ).images[0]
                images.append(img)
                if len(images) >= max_images:
                    break

        return images

    return decode_latents


@torch.no_grad()
def main():
    args = parse_args()
    # Load tokens
    data_dir = Path(args.token_dir)

    token_dataset = RawTokenDataset(data_dir, 1, filter_interrupts=False)
    video_data = token_dataset.data
    metadata = token_dataset.metadata

    images = decode_latents_wrapper(
        args.pretrained_model_name_or_path, args.unet_checkpoint_path,
        max_images=30
    )(video_data[args.offset::args.stride])
    output_gif_path = args.token_dir + f'/generated_offset{args.offset}.gif'
    export_to_gif(images, output_gif_path, args.fps)
    print(f'Saved to {output_gif_path}')

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

        output_comic_path = data_dir / f"generated_comic_offset{args.offset}.png"
        plt.savefig(output_comic_path, bbox_inches="tight")
        plt.close()
        print(f"Saved to {output_comic_path}")


if __name__ == "__main__":
    main()
