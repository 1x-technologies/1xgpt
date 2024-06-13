import json
import os

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset


class RawTokenDataset(TorchDataset):
    """ Loads raw uint16 tokens as memmap-backed array """
    def __init__(self, data_dir, window_size, stride=1, filter_interrupts=True):
        """
        Args:
            data_dir: directory with the same format as `data/train_v0` and `data/val_v0`.
                Notably, has `video.bin` and `metadata.json`
            window_size: number of frames per "video" sequence
            stride: frame skip
            filter_interrupts: Under 3% of frame sequences are the concatenation of two different clips.
                If filter_interrupts is True, will filter out these sequences using the segment ids.
        """
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        shape = (self.metadata["num_images"], self.metadata["s"], self.metadata["s"])
        video_tokens_path, states_tokens_path, action_tokens_path = [data_dir / f"{name}.bin" for name in ["video", "states", "actions"]]
        self.data = np.memmap(video_tokens_path, dtype=np.uint16, mode="r", shape=shape)
        self.window_size, self.stride = window_size, stride
        # Number of frames between the first and last frames of a video sequence (including one endpoint frame)
        self.video_len = (self.window_size - 1) * self.stride

        if filter_interrupts:
            seg_ids_path = data_dir / "segment_ids.bin"
            if not os.path.isfile(seg_ids_path):
                print("No `segment_ids.bin`, not filtering interrupted sequences")
                filter_interrupts = False
            else:
                seg_ids = np.memmap(seg_ids_path, dtype=np.int32, mode="r", shape=(self.metadata["num_images"],))

        self.valid_start_inds = []
        for start_ind in range(len(self.data) - self.video_len):
            # Assuming `seg_ids` is monotonically increasing, a sequence is interrupted
            # if the first and last frames have different segment ids.
            if not (filter_interrupts and seg_ids[start_ind] != seg_ids[start_ind + self.video_len]):
                self.valid_start_inds.append(start_ind)

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        """
        Returns a flattened sequence of tokens representing `self.window_size` frames,
        spaced `self.stride` apart.
        """
        start_ind = self.valid_start_inds[idx]
        x = torch.from_numpy((self.data[start_ind : start_ind + self.video_len + 1 : self.stride]).astype(np.int64))
        x = x.flatten()

        attention_mask = torch.ones_like(x)
        return {
            "input_ids": x,
            "labels": x,
            "attention_mask": attention_mask,
        }
