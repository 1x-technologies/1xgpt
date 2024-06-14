#!/usr/bin/env python3

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np

sys.path.append(os.getcwd())
from data import RawTokenDataset
from baselines.genie_world_model import LitWorldModel

STRIDE = 15


def parse_args():
    parser = argparse.ArgumentParser(description="Generates samples from GENIE model")
    parser.add_argument(
        "--val_data_dir", type=str, default="data/train_v0",
        help="A directory with video data, should have a `metadata.json` and `video.bin` We generate using the first frames of this dataset."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="data/genie_model/lightning_logs/version_0/checkpoints/epoch=0-step=97000.ckpt",
        help="A directory with the model weights and config.json."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/genie_generated", help="Directory to save generated outputs."
    )
    parser.add_argument(
        "--num_prompt_frames", type=int, default=8, help="The number of context frames."
    )
    parser.add_argument(
        "--window_size",  type=int, default=16, help="Will generate `window_size - num_prompt_frames` frames."
    )
    parser.add_argument(
        "--example_ind", type=int, default=0, help="The index in the dataset of the example to generate on."
    )
    parser.add_argument(
        "--teacher_force_time", action="store_true", help="If True, teacher-forces generation in time dimension."
    )
    parser.add_argument(
        "--single_pass", action="store_true", help="If True, visualizes argmax of single forward pass on fully masked inputs"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    assert args.num_prompt_frames <= args.window_size
    val_dataset = RawTokenDataset(args.val_data_dir, window_size=args.window_size, stride=STRIDE)
    latent_side_len = val_dataset.metadata["s"]

    # Get single example
    example_THW = val_dataset[args.example_ind]["input_ids"].reshape(1, args.window_size, latent_side_len, latent_side_len).to("cuda")
    
    # Laod the model checkpoint
    model = LitWorldModel.load_from_checkpoint(args.checkpoint, T=args.window_size, S=latent_side_len**2, image_vocab_size=1001).model
    model.eval()

    samples = []
    prompt_THW = example_THW.clone()
    prompt_THW[:, args.num_prompt_frames:] = model.image_mask_token

    if args.single_pass:
        # debugging viz: Teacher-forced across time, argmax 
        for timestep in range(args.num_prompt_frames, args.window_size):
            if args.teacher_force_time:
                prompt_THW = example_THW.clone()
                # Masked prediction for this timestep only, after which we provide ground-truth
                prompt_THW[:, timestep:] = model.image_mask_token
            logits_CTHW = model(prompt_THW)
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
            sample_HW, _ = model.maskgit_generate(prompt_THW, out_t=timestep, maskgit_steps=16, temperature=1.)
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
            "num_images": args.window_size * 2 - args.num_prompt_frames,
            "h": latent_side_len,
            "w": latent_side_len,
            "t": args.window_size,
        }, f)


if __name__ == "__main__":
    main()
