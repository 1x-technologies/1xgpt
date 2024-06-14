import os
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from absl import app, flags

from data import RawTokenDataset

sys.path.append(os.getcwd())
from baselines.genie_world_model import LitWorldModel

# DATA
flags.DEFINE_string("train_data_dir", 'data/train_v0', "Path to directory containing size.txt, video.bin, actions.bin, segment_ids.bin")
flags.DEFINE_string("val_data_dir", 'data/val_v0', "Path to directory containing validation tokens.")
flags.DEFINE_integer("batch_size", 4, "Batch size")
flags.DEFINE_integer("window_size", 16, "Number of input/output frames")
flags.DEFINE_integer("stride", 15, "How many frames to skip")
# TRAINING
flags.DEFINE_string("root_dir", 'data/genie_model', "Root directory for saving summaries and checkpoints")
flags.DEFINE_string("restore_ckpt", None, "Path to checkpoint to restore from")
flags.DEFINE_string("name", None, "Experiment name")

FLAGS = flags.FLAGS


def train(_):
    # tokenizer 
    T = FLAGS.window_size
    train_dataset = RawTokenDataset(FLAGS.train_data_dir, window_size=T, stride=FLAGS.stride)
    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=4,
        shuffle=True,
    )
    if FLAGS.val_data_dir:
        val_dataset = RawTokenDataset(FLAGS.val_data_dir, window_size=T, stride=FLAGS.stride)
        val_loader = DataLoader(
            val_dataset,
            batch_size=FLAGS.batch_size,
            num_workers=4,
        )
    else:
        val_loader = None
    with open(Path(FLAGS.train_data_dir) / "metadata.json") as f:
        metadata = json.load(f)
        s = metadata["s"]

    # Reserve a new MASK token value
    model = LitWorldModel(T=T, S=s*s, image_vocab_size=metadata['vocab_size'] + 1)
    
    # This method of restoring allows overriding lr_schedule / optimizer vars
    if FLAGS.restore_ckpt:
        checkpoint = torch.load(FLAGS.restore_ckpt, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])

    # Save every 1k steps
    steps_checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_train_steps=500,
        verbose=True,
        monitor=None,
    )
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [steps_checkpoint_callback, lr_monitor]
    
    root_dir = f"{FLAGS.root_dir}/{FLAGS.name}" if FLAGS.name else FLAGS.root_dir
    trainer = L.Trainer(
        max_epochs=100,
        default_root_dir=root_dir,
        log_every_n_steps=1,
        accelerator="gpu",
        precision="bf16-mixed",
        accumulate_grad_batches=4,
        callbacks=callbacks,
        gradient_clip_val=1.,
        val_check_interval=500,
        detect_anomaly=True,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    app.run(train)
