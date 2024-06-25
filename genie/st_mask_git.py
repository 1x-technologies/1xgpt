import math
from functools import partial

import mup
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from tqdm import tqdm
from transformers.utils import ModelOutput

from genie.config import GenieConfig
from genie.st_transformer import STTransformerDecoder


class STMaskGIT(nn.Module, PyTorchModelHubMixin):
    # Next-Token prediction as done in https://arxiv.org/pdf/2402.15391.pdf
    def __init__(self, config: GenieConfig):
        super().__init__()
        self.config = config
        self.h = self.w = round(math.sqrt(self.config.S))
        assert self.h**2 == self.config.S, "Expected S to be square"

        total_vocab_size = config.image_vocab_size + 1  # additional mask token
        self.mask_token_id = config.image_vocab_size

        self.token_embed = nn.Embedding(total_vocab_size, config.d_model)
        self.decoder = STTransformerDecoder(config)
        self.pos_embed_TSC = torch.nn.Parameter(torch.zeros(1, config.T, config.S, config.d_model))
        self.out_x_proj = (mup.MuReadout if config.use_mup else nn.Linear)(config.d_model, total_vocab_size)  # TODO: image_vocab_size instead?
        # MuReadout instead of nn.Linear slows down compiled training?

        # Register forward hooks for logging/debugging
        # Also register buffers which will accumulate these statistics
        for st_block_idx, st_block in enumerate(self.decoder.layers):
            module_abbr = f"stblock{st_block_idx}_spatial"
            self.register_buffer(module_abbr, torch.zeros((2, 2), dtype=torch.long), persistent=False)
            hook = partial(self.log_max_attn_values, module_abbr=module_abbr)
            st_block.spatial_attn.register_forward_hook(hook)
            module_abbr = f"stblock{st_block_idx}_temporal"
            self.register_buffer(module_abbr, torch.zeros((2, 2), dtype=torch.long), persistent=False)
            hook = partial(self.log_max_attn_values, module_abbr=module_abbr)
            st_block.temporal_attn.register_forward_hook(hook)

    def pred_tokens(self, x_THW):  # TODO: customize collator instead, unify forward
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
        logits_CTHW = self.pred_tokens(prompt_THW)
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
            # use cosine mask scheduling function, n is how many of frame out_t to mask
            n = math.ceil(math.cos((maskgit_t + 1) / maskgit_steps * (math.pi / 2)) * size)

            # set the n pixels with the smallest confidence to mask_token
            least_confident_tokens = torch.argsort(confidence, dim=1)
            # unmask the (L - n) most confident tokens
            unmasked.scatter_(1, least_confident_tokens[:, n:], True)
            sample.scatter_(1, least_confident_tokens[:, :n], self.mask_token_id)
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

    def generate(self, input_ids, attention_mask, max_new_tokens, min_new_tokens=None):
        """
        Args designed to match the format of Llama.
        We ignore `attention_mask`, and use `max_new_tokens` to determine the number of frames to generate.
        """
        assert min_new_tokens in (None, max_new_tokens), \
            "Expecting `min_new_tokens`, if specified, to match `max_new_tokens`."

        assert max_new_tokens % self.config.S == 0, "Expecting `max_new_tokens` to be a multiple of `self.config.S`."
        num_new_frames = max_new_tokens // self.config.S

        inputs_THW = rearrange(input_ids.clone(), "b (t h w) -> b t h w", h=self.h, w=self.w)
        inputs_masked_THW = torch.cat([
            inputs_THW,
            torch.full((input_ids.size(0), num_new_frames, self.h, self.w),
                       self.mask_token_id, dtype=torch.long, device=input_ids.device)
        ], dim=1)

        for timestep in range(inputs_THW.size(1), inputs_THW.size(1) + num_new_frames):
            sample_HW, _ = self.maskgit_generate(inputs_masked_THW, timestep, single_pass=True)
            inputs_masked_THW[:, timestep] = sample_HW

        return rearrange(inputs_masked_THW, "B T H W -> B (T H W)")

    @staticmethod
    def init_mask(prompt_THW):
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
        single_pass=False,
    ):
        # assume we have pre-masked z2...zT with all masks
        assert out_t, "maskgit_generate requires out_t > 0"
        assert torch.all(prompt_THW[:, out_t:] == self.mask_token_id), \
            f"when generating z{out_t}, frames {out_t} and later must be masked"

        # this will be modified by maskgit_generate_step
        unmasked = self.init_mask(prompt_THW)
        # unmasked are modified in place on each iteration of this loop
        logits_CHW = None
        if single_pass:
            logits_CTHW = self.pred_tokens(prompt_THW)
            logits_CHW = logits_CTHW[:, :-1, out_t]
            sample_HW = logits_CHW.argmax(dim=1)
            prompt_THW[:, out_t] = sample_HW
        else:
            for step in tqdm(range(maskgit_steps)):
                sample_HW, logits_CHW = self.maskgit_generate_step(
                    prompt_THW, out_t, step, unmasked, logits_CHW, maskgit_steps, temperature)
                # feed back to iteratively decode
                prompt_THW[:, out_t] = sample_HW
        # Return the final sample and logits
        return sample_HW, logits_CHW

    def log_max_attn_values(self, module, input, output, module_abbr):
        # TODO: Currently this seems to be the min/max after taking mean over batch dim, and after out_proj
        # Better to do actual attention matrix, but maybe hard to do with flash attention?
        # See if intermediate values are exploding
        if self.training:
            module_buffer = self.get_buffer(module_abbr)
            # module_buffer += torch.tensor([
            #     [, 0],
            #     [, 0]
            # ], device=module_buffer.device)
            # breakpoint()
            # x = torch.mean(output, dim=0)
            # # self., torch.max(x).float(), rank_zero_only=True)
            # self.log(f"min_qkv/{module_abbr}", torch.min(x).float(), rank_zero_only=True)

    def compute_loss(self, logits, x_targets, relevant_mask_THW):
        # Video token prediction
        x_output = logits[:, :, 1:]
        x_targets = x_targets[:, 1:]
        loss_THW = F.cross_entropy(x_output, x_targets, reduction="none")
        # self.log(f"img_loss/{split}", loss_THW.mean(), rank_zero_only=True)
        # acc = accuracy(x_output, x_targets)
        # self.log(f"img_acc/{split}", acc, add_dataloader_idx=False, rank_zero_only=True)
        # multiply loss values by mask instead of indexing them in compute_loss, more computationally
        # efficient. Compute the mean masked error.
        relevant_loss = torch.sum(loss_THW * relevant_mask_THW) / torch.sum(relevant_mask_THW)
        # self.log(f"masked_img_loss/{split}", relevant_loss, prog_bar=True, rank_zero_only=True)
        relevant_acc = torch.sum((x_output.argmax(dim=1) == x_targets) * relevant_mask_THW).float() / torch.sum(relevant_mask_THW)
        # self.log(f"masked_img_acc/{split}", acc_THW, prog_bar=True, rank_zero_only=True)

        # only optimize on the masked/noised logits?
        return relevant_loss, relevant_acc
        # return loss_THW.mean()

    def forward(self, input_ids, **kwargs):
        x = rearrange(input_ids, "b (t h w) -> b t h w", t=self.config.T, h=self.h, w=self.w)

        # we don't need to track gradients through these operations
        with torch.no_grad():
            # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
            # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1
            # Strictly speaking we don't need to predict the first
            x_THW = x.clone()

            # As done in Copilot-4D paper, add random noise sampled with a random rate between 0-20%
            r = torch.rand(x_THW.size(), device=x_THW.device)
            u01 = torch.rand((), device=x_THW.device)
            random_patches_mask = r < self.config.max_random_token_rate * u01
            random_values = torch.randint(low=0, high=self.config.image_vocab_size + 1, size=x_THW.size(),
                                          dtype=torch.long, device=x_THW.device)
            x_THW[random_patches_mask] = random_values[random_patches_mask]

            x_THW_view = x_THW[:, 1:]

            # per-minibatch, per-frame masking probability
            mask_prob_T = self.config.min_mask_rate + (self.config.max_mask_rate - self.config.min_mask_rate) \
                * torch.rand(x_THW_view.size(0), x_THW_view.size(1), device=x.device)
            r = torch.rand(x_THW_view.size(), device=x_THW_view.device)
            mask = r < mask_prob_T.unsqueeze(-1).unsqueeze(-1)
            # masking the view also masks x_TS
            x_THW_view[mask] = self.mask_token_id

        # Record the loss over masked tokens only to make it more comparable to LLM baselines
        is_masked = mask | random_patches_mask[:, 1:]

        # x_THW is for z0,...,zT while x_targets is z1,...,zT
        logits = self.pred_tokens(x_THW)
        relevant_loss, relevant_acc = self.compute_loss(logits, x, is_masked)

        return ModelOutput(loss=relevant_loss, logits=rearrange(logits, "B C T H W -> B (T H W) C"),
                           acc=relevant_acc)

    def init_weights(self):
        """ Works with and without muP. """
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module.weight, "infshape"):  # muP
                    mup.normal_(module.weight, mean=0.0, std=std)
                else:
                    module.weight.data.normal_(mean=0.0, std=std)

                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()


def accuracy(logits, targets):
    _, preds = torch.max(logits, dim=1)
    return torch.sum(preds == targets) / logits.shape[0]
