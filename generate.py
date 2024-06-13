#!/usr/bin/env python3

import argparse
import json

from data import RawTokenDataset
import torch
from transformers import AutoModelForCausalLM
import numpy as np
from pathlib import Path


STRIDE = 15


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--val_data_dir", type=str, default="data/val_v0",
        help="A directory with video data, should have a `metadata.json` and `video.bin`."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="data/model_checkpt",
        help="A directory with the model weights and config.json."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/generated", help="Directory to save generated outputs."
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

    return parser.parse_args()


def main():
    """
    In `output_dir / video.bin`, saves (back to back):
        - `num_prompt_frames` of prompt frames.
        - `window_size - num_prompt_frames` of autoregressively predicted frames given the prompt.
        - `window_size - num_prompt_frames` of ground truth frames corresponding to the predicted frames.
    """
    args = parse_args()
    assert args.num_prompt_frames <= args.window_size
    val_dataset = RawTokenDataset(args.val_data_dir, window_size=args.window_size, stride=STRIDE)
    latent_side_len = val_dataset.metadata["s"]

    # Note: ignoring attention mask
    reshaped_input_ids = val_dataset[args.example_ind]["input_ids"].reshape(args.window_size, -1).to("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to('cuda')
    model.eval()

    num_new_tokens = latent_side_len**2 * (args.window_size - args.num_prompt_frames)
    prompt_input_ids = reshaped_input_ids[:args.num_prompt_frames].reshape(1, -1)
    outputs = model.generate(input_ids=prompt_input_ids, attention_mask=torch.ones_like(prompt_input_ids),
                             max_new_tokens=num_new_tokens, min_new_tokens=num_new_tokens)

    outputs = torch.cat([outputs, reshaped_input_ids[:, args.num_prompt_frames:].reshape(1, -1)], dim=1)

    # write to output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs.cpu().numpy().astype(np.uint16).tofile(output_dir / "video.bin")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(vars(args) | val_dataset.metadata | {
            "num_images": args.window_size * 2 - args.num_prompt_frames,
            "s": latent_side_len,
        }, f)


if __name__ == "__main__":
    main()
