import math
from functools import partial

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from tqdm import tqdm

from genie.st_transformer import STTransformerDecoder


def accuracy(logits, targets):
    _, preds = torch.max(logits, dim=1)
    return torch.sum(preds == targets) / logits.shape[0]


class STWorldModel(nn.Module):
    # Next-Token prediction as done in https://arxiv.org/pdf/2402.15391.pdf
    def __init__(self, T, S, image_vocab_size, num_layers=8, num_heads=16, d_model=1024):
        super().__init__()
        self.image_mask_token = image_vocab_size - 1
        self.token_embed = nn.Embedding(image_vocab_size, d_model)
        self.decoder = STTransformerDecoder(num_layers=num_layers, dim=d_model, num_heads=num_heads)
        self.pos_embed_TSC = torch.nn.Parameter(torch.zeros(1, T, S, d_model))
        self.out_x_proj = nn.Linear(d_model, image_vocab_size)

    def forward(self, x_THW):
        T, H, W = x_THW.size(1), x_THW.size(2), x_THW.size(3)
        x_TS = rearrange(x_THW, "B T H W -> B T (H W)")
        x_TSC = self.token_embed(x_TS)

        # additive embeddings, using the same vocab space
        x_TSC = self.decoder(x_TSC + self.pos_embed_TSC)
        x_next_TSC = self.out_x_proj(x_TSC)
        x_CTHW = rearrange(x_next_TSC, "B T (H W) C -> B C T H W", H=H, W=W)
        return x_CTHW

    @torch.no_grad()
    def maskgit_generate_step(
        self,
        prompt_THW,
        out_t,
        maskgit_t,
        unmasked,
        prev_logits_CHW=None,
        maskgit_steps=25,
        temperature=1.,
    ):
        # Perform a single maskgit step (cosine schedule), updating unmasked in-place
        T, H, W = prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        logits_CTHW = self(prompt_THW)
        # Logits, not probs, among the non-mask classes
        # probs for the frame we care about
        probs = rearrange(logits_CTHW[:, :, out_t], "B C H W -> (B H W) C")
        unmasked_probs = probs[:, :-1]
        dist = torch.distributions.categorical.Categorical(logits=unmasked_probs / temperature)

        prev_unmasked = unmasked.clone()
        prev_img_flat = rearrange(prompt_THW[:, out_t], "B H W -> B (H W)")

        # logits for the frame we are generating
        if maskgit_t > 0:
            frame_prev_logits_flat = rearrange(prev_logits_CHW, "B C H W -> (B H W) C")

        # Predicted samples from new iteration
        sample = dist.sample()
        size = H * W
        # skip masking for last maskgit step
        if maskgit_t != maskgit_steps - 1:
            confidence = torch.gather(unmasked_probs, 1, sample[:, None])
            sample = sample.reshape(-1, size)
            confidence = confidence.reshape(-1, size)
            confidence[unmasked] = torch.inf
            # use cosine maks scheduling function
            n = math.ceil(
                math.cos((maskgit_t + 1) / maskgit_steps * (math.pi / 2)) * size)  # how many of frame out_t to mask
            # n = [50, 100, 150, 200, 0][maskgit_t]
            # set the n pixels with the smallest confidence to mask_token
            least_confident_tokens = torch.argsort(confidence, dim=1)
            # unmask the (L - n) most confident tokens
            unmasked.scatter_(1, least_confident_tokens[:, n:], True)
            sample.scatter_(1, least_confident_tokens[:, :n], self.image_mask_token)
        else:
            sample = sample.reshape(-1, size)

        # copy previously unmasked values from prompt input into sample
        sample[prev_unmasked] = prev_img_flat[prev_unmasked]
        if maskgit_t > 0:
            # copy previous unmasked values from previous logits into sample
            prev_unmasked_flat = prev_unmasked.flatten()
            unmasked_probs[prev_unmasked_flat] = frame_prev_logits_flat[prev_unmasked_flat]

        logits_CHW = rearrange(unmasked_probs, "(B H W) C -> B C H W", H=H, W=W)
        sample_HW = sample.reshape(-1, H, W)
        return sample_HW, logits_CHW

    def init_mask(self, prompt_THW):
        # since we generate 1 image at a time, the mask should be for a single frame, not across all frames.
        T, H, W = prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        unmasked = torch.zeros(prompt_THW.size(0), H * W, dtype=torch.bool, device=prompt_THW.device)
        return unmasked

    @torch.no_grad()
    def maskgit_generate(
        self,
        prompt_THW,
        out_t,
        maskgit_steps=8,
        temperature=1.,
    ):
        # assume we have pre-masked z2...zT with all masks
        assert out_t, "maskgit_generate requires out_t > 0"
        assert torch.all(prompt_THW[:,out_t:] == self.image_mask_token), \
            f"when generating z{out_t}, frames {out_t} and later must be masked"

        # this will be modified by maskgit_generate_step
        unmasked = self.init_mask(prompt_THW)
        # unmasked are modified in place on each iteration of this loop
        logits_CHW = None
        for step in tqdm(range(maskgit_steps)):
            sample_HW, logits_CHW = self.maskgit_generate_step(
                prompt_THW, out_t, step, unmasked, logits_CHW, maskgit_steps, temperature)
            # feed back to iteratively decode
            prompt_THW[:, out_t] = sample_HW
        # Return the final sample and logits
        return sample_HW, logits_CHW


class LitWorldModel(L.LightningModule, PyTorchModelHubMixin):
    def __init__(self,
        T,
        S,
        image_vocab_size,
        min_mask_rate=0.5,
        max_mask_rate=1.0,
        num_layers=8,
        num_heads=16,
        d_model=1024
    ):
        # T -- temporal sequence length, e.g.16
        # S -- spatial sequence length, e.g. 20x20 = 400
        super().__init__()
        self.model = STWorldModel(T, S, image_vocab_size, num_layers, num_heads, d_model)
        self.T = T
        self.h = self.w = int(math.sqrt(S))
        self.image_vocab_size = image_vocab_size

        # MaskGIT params
        self.min_mask_rate = min_mask_rate
        self.max_mask_rate = max_mask_rate
        self.max_random_token_rate = 0.1  # Probability of corrupting remaining unmasked tokens

        # Register forward hooks for logging/debugging
        for st_block_idx, st_block in enumerate(self.model.decoder.layers):
            name = f"stblock{st_block_idx}_spatial"
            hook = partial(self.log_max_attn_values, suffix=name)
            st_block.spatial_attn.register_forward_hook(hook)
            name = f"stblock{st_block_idx}_temporal"
            hook = partial(self.log_max_attn_values, suffix=name)
            st_block.temporal_attn.register_forward_hook(hook)

    def log_max_attn_values(self, module, input, output, suffix):
        # See if intermediate values are exploding
        if self.training:
            x = torch.mean(output, dim=0)
            self.log(f"max_qkv/{suffix}", torch.max(x).float(), rank_zero_only=True)
            self.log(f"min_qkv/{suffix}", torch.min(x).float(), rank_zero_only=True)

    def compute_loss(self, x_THW, x_targets, batch_idx, split, is_masked_THW):
        # x_THW is for z0,...,zT while x_targets is z1,...,zT
        x_output = self.model(x_THW)

        # Video token prediction
        x_output = x_output[:, :, 1:]
        x_targets = x_targets[:, 1:]
        # loss = F.cross_entropy(x_output, x_targets)
        loss_masked_THW = F.cross_entropy(x_output, x_targets, reduction="none")
        self.log(f"img_loss/{split}", loss_masked_THW.mean(), rank_zero_only=True)
        acc = accuracy(x_output, x_targets)
        self.log(f"img_acc/{split}", acc, add_dataloader_idx=False, rank_zero_only=True)
        # multiply loss values by mask instead of indexing them in compute_loss, more computationally
        # efficient. Compute the mean masked error.
        loss_masked_tokens = torch.sum(loss_masked_THW * is_masked_THW) / torch.sum(is_masked_THW)
        self.log(f"masked_img_loss/{split}", loss_masked_tokens, prog_bar=True, rank_zero_only=True)
        acc_THW = torch.sum((x_output.argmax(dim=1) == x_targets) * is_masked_THW).float() / torch.sum(is_masked_THW)
        self.log(f"masked_img_acc/{split}", acc_THW, prog_bar=True, rank_zero_only=True)

        # only optimize on the masked/noised logits?
        return loss_masked_tokens

    def training_step(self, batch, batch_idx, split="train"):
        x = rearrange(batch['input_ids'], "b (t h w) -> b t h w", t=self.T, h=self.h, w=self.w)

        # we don't need to track gradients through these operations
        with torch.no_grad():
            # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
            # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1
            # Strictly speaking we don't need to predict the first
            x_THW = x.clone()

            # As done in Copilot-4D paper, add random noise sampled with a random rate between 0-20%
            r = torch.rand(x_THW.size(), device=x_THW.device)
            u01 = torch.rand((), device=x_THW.device)
            random_patches_mask = r < self.max_random_token_rate * u01
            random_values = torch.randint(low=0, high=self.image_vocab_size, size=x_THW.size(), dtype=torch.long,
                                          device=x_THW.device)
            x_THW[random_patches_mask] = random_values[random_patches_mask]

            x_THW_view = x_THW[:, 1:]

            # per-minibatch, per-frame masking probability
            mask_prob_T = self.min_mask_rate + (self.max_mask_rate - self.min_mask_rate) * torch.rand(
                x_THW_view.size(0), x_THW_view.size(1), device=x.device)
            r = torch.rand(x_THW_view.size(), device=x_THW_view.device)
            mask = r < mask_prob_T.unsqueeze(-1).unsqueeze(-1)
            # masking the view also masks x_TS
            x_THW_view[mask] = self.model.image_mask_token

        # Record the loss over masked tokens only to make it more comparable to LLM baselines
        is_masked = mask | random_patches_mask[:, 1:]
        loss = self.compute_loss(x_THW, x, batch_idx, split, is_masked)

        return loss

    def validation_step(self, batch, batch_idx):
        # Compute same metrics as done in train
        return self.training_step(batch, batch_idx, split="val")

    def configure_optimizers(self):
        lr = 1e-4
        opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=1e-4)

        scheduler = get_scheduler(
            name="cosine",
            optimizer=opt,
            num_warmup_steps=5_000,
            num_training_steps=300_000,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @classmethod
    def load_model(cls, hf_checkpoint=None, lightning_checkpoint=None, **kwargs):
        """
        kwargs: extra arguments passed to `LitWorldModel` when loading from `lightning_checkpoint`, like
            `num_layers`, `num_heads`
        """
        assert (hf_checkpoint is not None) ^ (lightning_checkpoint is not None), \
            "Exactly one of `hf_checkpoint` and `lightning_checkpoint` should be provided."

        if hf_checkpoint is not None:
            return LitWorldModel.from_pretrained(hf_checkpoint).model
        else:
            return LitWorldModel.load_from_checkpoint(
                lightning_checkpoint, **kwargs
            ).model
