import json
from pathlib import Path

import lightning as L
import torch
from absl import app, flags
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from data import RawTokenDataset
from genie.genie_world_model import LitWorldModel

# DATA
flags.DEFINE_string("train_data_dir", "data/train_v0",
                    "Path to directory containing size.txt, video.bin, actions.bin, segment_ids.bin")
flags.DEFINE_string("val_data_dir", "data/val_v0", "Path to directory containing validation tokens.")
flags.DEFINE_integer("batch_size", 2, "Batch size")
flags.DEFINE_integer("window_size", 16, "Number of input/output frames")
flags.DEFINE_integer("stride", 15, "How many frames to skip")
# TRAINING
flags.DEFINE_string("root_dir", "data/genie_model", "Root directory for saving summaries and checkpoints")
flags.DEFINE_string("restore_ckpt", None, "Path to checkpoint to restore from")
flags.DEFINE_string("name", None, "Experiment name")
flags.DEFINE_integer("num_layers", default=12, help="Num hidden layers")
flags.DEFINE_integer("num_heads", default=16, help="Num attention heads")
flags.DEFINE_integer("d_model", default=1024, help="Hidden size")


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
        generator=torch.Generator().manual_seed(42),
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
    model = LitWorldModel(T=T, S=s * s, image_vocab_size=metadata["vocab_size"] + 1, num_layers=FLAGS.num_layers,
                          num_heads=FLAGS.num_heads, d_model=FLAGS.d_model)

    # This method of restoring allows overriding lr_schedule / optimizer vars
    if FLAGS.restore_ckpt:
        checkpoint = torch.load(FLAGS.restore_ckpt, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        # Does not restore dataloader, lr scheduler yet

    # Save every 5k steps
    steps_checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_train_steps=5000,
        verbose=True,
        monitor=None,
    )
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [steps_checkpoint_callback, lr_monitor]
    wandb_logger = L.pytorch.loggers.WandbLogger(project="1XGPT_20x20_noconv_noaugment", log_model=False)

    root_dir = f"{FLAGS.root_dir}/{FLAGS.name}" if FLAGS.name else FLAGS.root_dir
    trainer = L.Trainer(
        max_epochs=1,
        default_root_dir=root_dir,
        log_every_n_steps=1,
        accelerator="gpu",
        precision="bf16-mixed",
        accumulate_grad_batches=2,
        callbacks=callbacks,
        gradient_clip_val=1.,
        val_check_interval=500,
        logger=wandb_logger,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    app.run(train)
