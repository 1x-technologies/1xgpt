import argparse
import contextlib
import logging
import math
import os
import time

import matplotlib
import mup
import numpy as np
import torch
import torchvision.transforms.functional as transforms_f
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from einops import rearrange
from lpips import lpips
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    default_data_collator,
    get_scheduler,
)

from data import RawTokenDataset, get_maskgit_collator
from eval_utils import decode_tokens, compute_lpips
from genie.st_mask_git import GenieConfig, STMaskGIT
# from llama.config import LlamaConfig1X
# from llama.modeling_llama_mup import LlamaForCausalLM
from visualize import decode_latents_wrapper

matplotlib.use("Agg")
from matplotlib import pyplot as plt

torch.set_float32_matmul_precision("medium")
logger = get_logger(__name__)


def parse_args():
    # parser = argparse.ArgumentParser(description="Train a MaskGIT or Llama-style LLM on video generation.")
    parser = argparse.ArgumentParser(description="Train a spatial-temporal MaskGIT-style model on video generation.")

    # Data
    parser.add_argument(
        "--train_data_dir", type=str, default="data/train_v1.0",
        help="Directory containing tokenized data, should have a `video.bin`, `metadata.json` and `segment_ids.json`."
    )
    parser.add_argument(
        "--val_data_dir", type=str, default="data/val_v1.0",
        help="Directory containing tokenized data, should have a `video.bin`, `metadata.json` and `segment_ids.json`."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Number of frames to in a sequence.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Difference in frame count between consecutive frames in a sequence.",
    )
    parser.add_argument(
        "--filter_overlaps",
        action="store_true",
        help=(
            "Whether to filter repeated frames in the train dataset (`filter_overlaps` always true for the val set). "
            "Filtering essentially makes the training dataset less correlated but ~16x smaller, "
            "see the `filter_overlaps` argument in `RawTokenDataset` for details.")
        ,
    )

    # Model
    parser.add_argument(
        "--llama_config",
        type=str,
        help="`transformers.LlamaConfig` json. "
             "E.g. https://huggingface.co/1x-technologies/Llama_1B_v0/blob/main/config.json",
    )
    parser.add_argument(
        "--genie_config",
        type=str,
        help="GenieConfig json."
    ),
    parser.add_argument(
        "--warmstart_path",
        type=str,
        default=None,
        help="A path to a checkpoint to warmstart a model from, possibly not trained on the same dataset, "
             "will resize embeddings if needed.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    # Training
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=1e10,
        help="Only evaluate on `max_eval_steps` batches of validation data per process, faster.",
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=1000,
        help="Eval every N training steps.",
    )
    parser.add_argument(
        "--vis_every_n_steps",
        type=int,
        default=1000,
        help="Visualize every N training steps.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "custom_cosine"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Threshold to clip gradients.",
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.0,
        help="Attention dropout prob.",
    )
    parser.add_argument(
        "--adam_beta_1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta_2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=1e-8,
    )

    # Misc
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the model checkpoints.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="1000",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--overfit_first_batch",
        action="store_true",
        help=(
            "Debug option that trains and validates on only the first batch of the training dataset."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="The integration to report the results and logs to.",
    )
    parser.add_argument(
        "--mu_transfer",
        action="store_true",
        help="If specified, will train with mu transfer reparametrizations. Only supports Llama models."
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="If specified, will not compile the model."
    )

    args = parser.parse_args()

    return args


def save_checkpoint(model, accelerator, args, filename):
    """
    Args:
        model:
        accelerator:
        args:
        filename: `save_path = os.path.join(args.output_dir, filename)`
    """
    unwrapped_model = accelerator.unwrap_model(model)
    save_path = os.path.join(args.output_dir, filename)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            save_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        accelerator.save_state(save_path)


@torch.no_grad()
def visualize(accelerator, model, dataloader, window_size, metrics_prefix="eval", max_steps=1):
    """
    Visualizes model's autoregressive generation outputs, logged to wandb.

    metrics_prefix: each metric is logged as f"{metrics_prefix}_{metric_key}". Also used in name of wandb figure.
    """
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    metadata = dataloader.dataset.metadata
    decode_latents = decode_latents_wrapper()  # re-initializing every time to save memory
    if accelerator.is_main_process:
        lpips_alex = lpips.LPIPS(net="alex")  # Calculate LPIPS w/ AlexNet, the fastest option
        metrics = {"pred_lpips": []}

    latent_side_len = metadata["s"]

    unwrapped_model.eval()
    for step, batch in enumerate(dataloader):
        # Note: hardcoding 4 image cap for faster inference on small models
        reshaped_labels = rearrange(batch["labels"][:4], "b (t s) -> b t s", t=window_size).to(accelerator.device)  # `s` is really `(h, w)`

        num_prompt_frames = window_size // 2  # hardcoding half of frames for context
        num_new_tokens = latent_side_len ** 2 * (window_size - num_prompt_frames)
        prompt_input_ids = rearrange(reshaped_labels[:, :num_prompt_frames], "b t s -> b (t s)")
        outputs = unwrapped_model.generate(input_ids=prompt_input_ids, attention_mask=torch.ones_like(prompt_input_ids),
                                           max_new_tokens=num_new_tokens, min_new_tokens=num_new_tokens)
        output_tokens = rearrange(outputs, "b (t h w) -> b t h w", t=window_size,
                                  h=latent_side_len, w=latent_side_len)
        gtruth_tokens = rearrange(reshaped_labels[:, num_prompt_frames:], "b t (h w) -> b t h w",
                                  h=latent_side_len, w=latent_side_len)

        decoded_output = decode_tokens(output_tokens.cpu(), decode_latents)
        decoded_gtruth = decode_tokens(gtruth_tokens.cpu(), decode_latents)

        decoded_output = accelerator.gather(decoded_output.to(accelerator.device)).cpu()
        decoded_gtruth = accelerator.gather(decoded_gtruth.to(accelerator.device)).cpu()

        if accelerator.is_main_process:
            exs_per_fig = 4
            for j in range(0, len(decoded_output), exs_per_fig):
                fig, axs = plt.subplots(nrows=2 * exs_per_fig, ncols=window_size, figsize=(3 * window_size, 3 * 2 * exs_per_fig))
                # If len(decoded_output) is not a multiple of 4, make sure to truncate properly
                for k in range(min(exs_per_fig, len(decoded_output) - j)):
                    for i in range(num_prompt_frames):
                        for ax in (axs[k * 2, i], axs[k * 2 + 1, i]):
                            ax.imshow(transforms_f.to_pil_image(decoded_output[j + k, i]))
                            ax.set_title("Context")
                            ax.axis("off")

                    for i in range(num_prompt_frames, window_size):
                        axs[k * 2, i].imshow(transforms_f.to_pil_image(decoded_gtruth[j + k, i - num_prompt_frames]))
                        axs[k * 2, i].set_title("Ground truth")
                        axs[k * 2 + 1, i].imshow(transforms_f.to_pil_image(decoded_output[j + k, i]))
                        axs[k * 2 + 1, i].set_title("Prediction")
                        for ax in axs[:, i]:
                            ax.axis("off")

                wandb_tracker = accelerator.get_tracker("wandb")
                wandb_tracker.log({f"vis_{metrics_prefix}_{j}": fig}, commit=False)
                plt.close(fig)

            metrics["pred_lpips"].extend(compute_lpips(decoded_gtruth,  # Note: not parallelizing right now
                                                       decoded_output[:, num_prompt_frames:], lpips_alex))

        if step + 1 >= max_steps:
            break

    unwrapped_model.train()
    if accelerator.is_main_process:
        metrics = {f"{metrics_prefix}_{key}": np.mean(val) for key, val in metrics.items() if len(val) > 0}

        print(f"{metrics=}")
        wandb_tracker = accelerator.get_tracker("wandb")
        wandb_tracker.log(metrics, commit=False)


def main():
    args = parse_args()
    assert (args.llama_config is not None) ^ (args.genie_config is not None), \
        "Exactly one of `llama_config` and `genie_config` should be set."

    # Manual gradient accumulation
    accelerator = Accelerator(gradient_accumulation_steps=1, log_with=args.report_to, project_dir=args.output_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    train_dataset = RawTokenDataset(args.train_data_dir, window_size=args.window_size,
                                    stride=args.stride, filter_overlaps=args.filter_overlaps)
    if not args.overfit_first_batch:
        eval_dataset = RawTokenDataset(args.val_data_dir, window_size=args.window_size,
                                       stride=args.stride, filter_overlaps=True)
    else:
        train_dataset.valid_start_inds = train_dataset.valid_start_inds[:args.per_device_train_batch_size
                                                                         * args.gradient_accumulation_steps
                                                                         * accelerator.num_processes]
        eval_dataset = train_dataset

    assert all(train_dataset.metadata[shared_key] == eval_dataset.metadata[shared_key]
               for shared_key in ("s", "vocab_size", "hz"))

    latent_side_len, vocab_size, hz = [train_dataset.metadata[key] for key in ("s", "vocab_size", "hz")]

    if args.llama_config is not None:
        raise NotImplementedError("Have not factorized Llama vocabulary.")
        # # rope_theta 500_000: https://arxiv.org/abs/2309.16039
        # config = LlamaConfig1X.from_pretrained(
        #     args.llama_config,
        #     vocab_size=vocab_size,
        #     pad_token_id=None,
        #     attention_dropout=args.attention_dropout,
        #     max_position_embeddings=latent_side_len**2 * args.window_size,
        #     rope_theta=500_000,
        #     use_mup=args.mu_transfer,
        #     _attn_implementation="flash_attention_2"
        # )
        #
        # if hasattr(config, "max_sequence_length"):  # now `max_position_embeddings`
        #     del config.max_sequence_length
        #
        # logger.info("Training new model from scratch")
        # logging.info(config)
        #
        # if args.warmstart_path is not None:
        #     raise NotImplementedError
        #
        #     # model = AutoModelForCausalLM.from_pretrained(args.warmstart_path,
        #     #                                              attn_implementation="flash_attention_2")
        #     # model.model.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        #     # model = model.to(dtype=torch.bfloat16)
        #
        # model = LlamaForCausalLM(config=config).to(dtype=torch.bfloat16)
        #
        # if args.mu_transfer:
        #     base_config = LlamaConfig1X(**(copy.deepcopy(vars(config)) | {
        #         "hidden_size": 512,
        #         "intermediate_size": 1024,
        #         "num_attention_heads": 8,
        #     }))
        #
        #     base_model = LlamaForCausalLM(config=base_config)
        #
        #     mup.set_base_shapes(model, base_model)
        #     model.apply(model._init_weights)  # Note: cannot simply call init_weights because `_is_hf_initialized` is already True
        # else:
        #     model = AutoModelForCausalLM.from_config(config,
        #                                              torch_dtype=torch.bfloat16,
        #                                              attn_implementation="flash_attention_2")
    else:
        config = GenieConfig.from_pretrained(args.genie_config)
        config.use_mup = args.mu_transfer  # Note: changing this may affect pre-trained model due to attn scaling
        config.image_vocab_size = vocab_size
        config.T = args.window_size
        config.S = latent_side_len**2
        model = STMaskGIT(config)

        if args.mu_transfer:
            model.set_mup_shapes(rescale_params=True)
            model.init_weights()  # might be unnecessary if `rescale_params` is True

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    opt_class = mup.MuAdamW if args.mu_transfer else torch.optim.AdamW
    optimizer = opt_class(optimizer_grouped_parameters, lr=args.learning_rate,
                          betas=(args.adam_beta_1, args.adam_beta_2), eps=args.adam_eps)

    # DataLoaders creation:
    collate_fn = default_data_collator if args.llama_config is not None else get_maskgit_collator(config)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size, num_workers=4, pin_memory=True,
    )

    # Shuffle eval dataset and then set shuffle=False on the dataloader.
    # Shuffling in the dataloader results in reshuffling with each iteration.
    eval_dataset.valid_start_inds = torch.tensor(eval_dataset.valid_start_inds)[
        torch.randperm(len(eval_dataset), generator=torch.Generator().manual_seed(0))
    ].tolist()

    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=collate_fn,
        batch_size=args.per_device_eval_batch_size, pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.lr_scheduler_type == "custom_cosine":  # decay to `end_ratio` of the peak learning rate
        def get_lr_wrapper(warmup_steps, max_steps, end_ratio=0.1):
            def get_lr(step):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps

                remaining_steps = max_steps - warmup_steps
                return ((1 + math.cos(math.pi * (step - warmup_steps) / remaining_steps)) / 2) * (
                        1 - end_ratio) + end_ratio
            return get_lr

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, get_lr_wrapper(args.num_warmup_steps * accelerator.num_processes,
                                      args.max_train_steps if overrode_max_train_steps
                                      else args.max_train_steps * accelerator.num_processes)
        )
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes,
        )

    # Enable gradient checkpointing to save memory
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False # incompatible with grad checkpointing

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    if not args.no_compile:
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.optimize_ddp = False  # https://github.com/pytorch/pytorch/issues/104674
        # TODO: https://github.com/pytorch/pytorch/issues/109774#issuecomment-2046633776
        model = torch.compile(model)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    experiment_config = vars(args) | vars(config)

    seq_len = latent_side_len**2 * args.window_size
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps \
                           * accelerator.num_processes
    experiment_config.update({
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_parameters_M": round(sum(p.numel() for p in model.parameters()) / 1e6),
        "seq_len": seq_len,
        "hz": hz / args.stride,
        "train_data_tokens": len(train_dataset) * seq_len,
        "effective_batch_size": effective_batch_size,
        "effective_batch_size_tokens": effective_batch_size * seq_len,
        "mixed_precision": accelerator.mixed_precision,
    })

    experiment_config["FLOPs_per_update_step"] = 6 * experiment_config["model_parameters"] \
                                                 * experiment_config["effective_batch_size_tokens"]

    accelerator.init_trackers(project_name="1XGPT_muP_MAGVIT2_v0", config=experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

        tied_weights = getattr(config, "tie_word_embeddings", False)
        accelerator.load_state(checkpoint_path, strict=not tied_weights)  # tied weights not saved so can't load strict, but also no need to tie again

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    loss_info = torch.zeros(2, device=accelerator.device)  # sum, count

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        _time = time.time()
        for step, batch in enumerate(active_dataloader):
            batch_size = batch["input_ids"].size(0)
            # Manual gradient accumulation because accelerator somehow taking a lot of memory
            is_update_step = (step + 1) % args.gradient_accumulation_steps == 0
            ctx_manager = contextlib.nullcontext() if is_update_step else accelerator.no_sync(model)

            with ctx_manager:
                outputs = model(**batch)
                loss = outputs.loss
                # print(f"{loss.item()=}")
                loss_info[0] += loss.detach() * batch_size
                loss_info[1] += batch_size

                accelerator.backward(loss / args.gradient_accumulation_steps)

            if not is_update_step:
                continue

            # Everything below only happens on update step

            if args.max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss_info = accelerator.reduce(loss_info)
            avg_train_loss = (loss_info[0] / loss_info[1]).item()  # sum / count
            loss_info *= 0  # reset sum and count
            try:
                perplexity = math.exp(avg_train_loss)
            except OverflowError:
                perplexity = float("inf")

            batch_time = time.time() - _time  # accumulated batch
            _time = time.time()
            accelerator.log(
                {
                    "train_perplexity": perplexity,
                    "train_loss": avg_train_loss,
                    "epoch": epoch,
                    "update_step": completed_steps,
                    "examples_processed": completed_steps * args.per_device_train_batch_size
                                          * args.gradient_accumulation_steps * accelerator.num_processes,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "flops": (completed_steps + 1) * experiment_config["FLOPs_per_update_step"],
                    "throughput_examples": experiment_config["effective_batch_size"] / batch_time,
                }, step=completed_steps)

            progress_bar.update(1)
            completed_steps += 1

            if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                save_checkpoint(model, accelerator, args, f"step_{completed_steps}")

            if completed_steps % args.eval_every_n_steps == 0:
                model.eval()

                eval_losses = []

                # Compute token-level accuracy (w/ teacher forcing)
                num_correct = 0
                num_total = 0
                for step, batch in enumerate(eval_dataloader):
                    batch_size = len(batch["input_ids"])  # Last batch might not be full
                    with torch.no_grad():
                        outputs = model(**batch)

                    loss = outputs.loss
                    eval_losses.append(accelerator.gather_for_metrics(loss.repeat(batch_size)))

                    if "acc" in outputs:  # TODO: don't reduce here
                        # `num_correct` and `num_total` actually track mean accuracy in this case.
                        num_correct += accelerator.reduce(outputs.acc, reduction="mean").item() * batch_size
                        num_total += batch_size
                    else:
                        shifted_preds = torch.argmax(outputs.logits[:, :-1, :], dim=-1)
                        shifted_labels = batch["labels"][:, 1:]
                        num_correct += accelerator.gather_for_metrics((shifted_preds == shifted_labels).sum()).sum().item()
                        num_total += accelerator.gather_for_metrics(torch.tensor(torch.numel(shifted_labels),
                                                                                 device=accelerator.device)).sum().item()
                    if step >= args.max_eval_steps:
                        break

                eval_losses = torch.cat(eval_losses)
                eval_loss = torch.mean(eval_losses).item()
                eval_teacher_acc = num_correct / num_total
                try:
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"{completed_steps=} {perplexity=} {eval_loss=} {eval_teacher_acc=}")

                accelerator.log(
                    {
                        "eval_perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "eval_teacher_acc": eval_teacher_acc,
                        "epoch": epoch,
                        "update_step": completed_steps,
                        "examples_processed": completed_steps * args.per_device_train_batch_size
                                              * args.gradient_accumulation_steps * accelerator.num_processes,
                        "flops": completed_steps * experiment_config["FLOPs_per_update_step"],
                    },
                    step=completed_steps,
                )

                # Switch back to train mode
                model.train()

            if completed_steps % args.vis_every_n_steps == 0:
                if not args.overfit_first_batch:  # val is train otherwise
                    visualize(accelerator, model, eval_dataloader, args.window_size, "val")

                visualize(accelerator, model, train_dataloader, args.window_size, "train")

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            save_checkpoint(model, accelerator, args, f"epoch_{epoch}")

    accelerator.end_training()
    save_checkpoint(model, accelerator, args, f"final_checkpt")


if __name__ == "__main__":
    main()
