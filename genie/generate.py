#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import mup
import torch
import numpy as np

sys.path.append(os.getcwd())
from data import RawTokenDataset
from genie.st_mask_git import STMaskGIT

STRIDE = 15


def parse_args():
    parser = argparse.ArgumentParser(description="Generates samples from GENIE model")
    parser.add_argument(
        "--val_data_dir", type=str, default="data/val_v0",
        help="A directory with video data, should have a `metadata.json` and `video.bin` We generate using the first frames of this dataset."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        help="Path to a HuggingFace-style checkpoint."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/genie_generated", help="Directory to save generated outputs."
    )
    parser.add_argument(
        "--num_prompt_frames", type=int, default=8, help="The number of context frames."
    )
    parser.add_argument(
        "--window_size", type=int, default=16, help="Will generate `window_size - num_prompt_frames` frames."
    )
    parser.add_argument(
        "--example_ind", type=int, default=0, help="The index in the dataset of the example to generate on."
    )
    parser.add_argument(
        "--teacher_force_time", action="store_true", help="If True, teacher-forces generation in time dimension."
    )
    parser.add_argument(
        "--single_pass", action="store_true",
        help="If True, takes argmax of single forward pass on fully masked inputs"
    )
    parser.add_argument(
        "--maskgit_steps", type=int, default=8, help="Number of maskgit sampling steps."
    )
    parser.add_argument(
        "--temperature", type=float, default=1., help="Sampling temperature."
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    assert args.num_prompt_frames <= args.window_size
    val_dataset = RawTokenDataset(args.val_data_dir, window_size=args.window_size, stride=STRIDE)
    latent_side_len = val_dataset.metadata["s"]

    # Get single example
    example_THW = val_dataset[args.example_ind]["input_ids"].reshape(1, args.window_size, latent_side_len,
                                                                     latent_side_len).to("cuda")

    # Load the model checkpoint
    model = STMaskGIT.from_pretrained(args.checkpoint_dir).to("cuda")

    base_config = model.config.shallow_copy()  # TODO: store with checkpoint
    base_config.num_heads = 8
    base_config.d_model = 256
    base_model = STMaskGIT(base_config)

    mup.set_base_shapes(model, base_model)

    model.eval()

    samples = []
    prompt_THW = example_THW.clone()
    prompt_THW[:, args.num_prompt_frames:] = model.mask_token_id

    if args.single_pass:
        # debugging viz: Teacher-forced across time, argmax 
        for timestep in range(args.num_prompt_frames, args.window_size):
            if args.teacher_force_time:
                prompt_THW = example_THW.clone()
                # Masked prediction for this timestep only, after which we provide ground-truth
                prompt_THW[:, timestep:] = model.mask_token_id
            logits_CTHW = model.pred_tokens(prompt_THW)
            sample_HW = logits_CTHW[:, :, timestep].argmax(dim=1)
            samples.append(sample_HW)
            if not args.teacher_force_time:
                # autoregressive, feed output back
                prompt_THW[:, timestep] = sample_HW
    else:
        for timestep in range(args.num_prompt_frames, args.window_size):
            # Teacher-forced, maskgit generation
            if args.teacher_force_time:
                prompt_THW = example_THW.clone()
                # Masked prediction for this timestep only, after which we provide ground-truth
                prompt_THW[:, timestep:] = model.image_mask_token

            sample_HW, _ = model.maskgit_generate(
                prompt_THW, out_t=timestep, maskgit_steps=args.maskgit_steps, temperature=args.temperature
            )

            samples.append(sample_HW)
            if not args.teacher_force_time:
                # autoregressive
                prompt_THW[:, timestep] = sample_HW

    outputs = torch.stack(samples, dim=1)
    # prepend prompt sequence
    outputs = torch.cat([example_THW[:, :args.num_prompt_frames], outputs], dim=1)

    # append ground-truth targets next to generated outputs for comic strip generation
    # [<prompt frames><predicted frames><ground truth frames>]
    outputs = torch.cat([outputs, example_THW[:, args.num_prompt_frames:]], dim=1)

    # write to output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs.cpu().numpy().astype(np.uint16).tofile(output_dir / "video.bin")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(vars(args) | val_dataset.metadata | {
            "num_images": outputs.shape[1],
            "h": latent_side_len,
            "w": latent_side_len,
            "t": args.window_size,
        }, f)


if __name__ == "__main__":
    main()
