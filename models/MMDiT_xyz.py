from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.MMDiT import MMDiT
from utils.train_utils import lengths_to_mask


class MMDiTXYZFinalLayer(nn.Module):
    """Decode non-overlapping spatiotemporal patch tokens to raw XYZ joints."""

    def __init__(
        self,
        hidden_size: int,
        output_channels: int,
        patch_size: tuple[int, int],
        patches_per_frame: int,
        joint_count: int,
    ) -> None:
        super().__init__()
        if patch_size[0] != 1:
            raise ValueError(
                "MMDiTXYZFinalLayer currently requires patch_size[0] == 1."
            )
        self.output_channels = output_channels
        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.joint_count = joint_count
        self.linear = nn.Linear(
            hidden_size,
            output_channels * patch_size[0] * patch_size[1],
            bias=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        if token_count % self.patches_per_frame != 0:
            raise ValueError(
                f"Token count {token_count} is not divisible by "
                f"patches_per_frame={self.patches_per_frame}."
            )
        frame_count = token_count // self.patches_per_frame
        patches = self.linear(tokens).reshape(
            batch_size,
            frame_count,
            self.patches_per_frame,
            self.output_channels,
            self.patch_size[1],
        )
        motion = patches.permute(0, 3, 1, 2, 4).reshape(
            batch_size,
            self.output_channels,
            frame_count,
            self.joint_count,
        )
        return motion


class MMDiT_xyz(MMDiT):
    """Multimodal MMDiT operating natively on [B, XYZ, time, joint]."""

    raw_joint_layout = True

    def __init__(
        self,
        input_dim: int,
        raw_joint_dim: int = 3,
        raw_joint_count: int = 22,
        patch_size: tuple[int, int] = (1, 22),
        stride_size: tuple[int, int] = (1, 22),
        **kwargs,
    ) -> None:
        if input_dim not in (raw_joint_dim, raw_joint_dim * raw_joint_count):
            raise ValueError(
                "MMDiT_xyz input_dim must be the XYZ channel count or flattened "
                f"raw dimension, got {input_dim}."
            )
        patch_size = tuple(int(value) for value in patch_size)
        stride_size = tuple(int(value) for value in stride_size)
        if patch_size[0] != 1 or stride_size[0] != 1:
            raise ValueError("MMDiT_xyz requires temporal patch and stride 1.")
        if patch_size != stride_size:
            raise ValueError("MMDiT_xyz currently requires non-overlapping patches.")
        if raw_joint_count % patch_size[1] != 0:
            raise ValueError(
                f"raw_joint_count={raw_joint_count} must be divisible by "
                f"joint patch size {patch_size[1]}."
            )

        super().__init__(input_dim=raw_joint_dim, **kwargs)
        self.raw_joint_dim = raw_joint_dim
        self.raw_joint_count = raw_joint_count
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.patches_per_frame = raw_joint_count // patch_size[1]

        self.x_embedder = nn.Conv2d(
            self.input_dim,
            self.latent_dim,
            kernel_size=self.patch_size,
            stride=self.stride_size,
            bias=True,
        )
        self.final_layer = MMDiTXYZFinalLayer(
            self.latent_dim,
            self.input_dim,
            self.patch_size,
            self.patches_per_frame,
            self.raw_joint_count,
        )
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _canonicalize_raw_motion(self, motion: torch.Tensor) -> torch.Tensor:
        if motion.dim() != 4:
            raise ValueError(
                "MMDiT_xyz motion must have four dimensions, "
                f"got {tuple(motion.shape)}."
            )
        if (
            motion.shape[1] == self.raw_joint_dim
            and motion.shape[3] == self.raw_joint_count
        ):
            return motion
        if motion.shape[-2:] == (self.raw_joint_count, self.raw_joint_dim):
            return motion.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            "MMDiT_xyz motion must have shape [B, 3, T, 22] or [B, T, 22, 3], "
            f"got {tuple(motion.shape)}."
        )

    def _sample_scalar_flow_time(self, target: torch.Tensor) -> torch.Tensor:
        transport = self.train_diffusion
        t0, t1 = transport.check_interval(transport.train_eps, transport.sample_eps)
        shape = (target.shape[0], 1)
        if self.flow_t_sampler == "logit":
            time = torch.sigmoid(
                torch.randn(shape, device=target.device) * 0.8 + self.flow_t_logit_mu
            )
            return torch.clamp(t0 + (t1 - t0) * time, min=t0, max=t1)
        return torch.rand(shape, device=target.device) * (t1 - t0) + t0

    def _sample_flow_training_state(self, target):
        return self._sample_scalar_flow_time(target), torch.randn_like(target), target

    def forward_step(
        self,
        x,
        t,
        conds,
        conds_mask,
        attention_mask,
    ):
        batch_size = x.shape[0]
        raw_t = t
        t_emb = self.t_embedder(t, dtype=x.dtype)
        t = t_emb

        conds = self.mask_cond(conds)
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        conds = self.y_embedder(conds)
        conds = rearrange(conds, "len batch dim -> batch len dim", batch=batch_size)

        x_valid = self.normalize_valid_mask(
            attention_mask,
            batch_size,
            x.shape[1],
            x.device,
            "attention_mask",
        )
        text_valid = self.normalize_valid_mask(
            conds_mask,
            batch_size,
            conds.shape[1],
            x.device,
            "conds_mask",
        )
        if self.y_token_refiner is not None:
            refiner_t = self.single_token_refiner_timestep(raw_t, x_valid)
            conds = self.y_token_refiner(conds, refiner_t, text_valid).to(
                dtype=conds.dtype
            )
        global_conds = self.build_global_conds(
            conds, text_valid, self.global_y_embedder, t.dtype
        )
        c_img = F.silu(t + global_conds)
        c_txt = self.masked_mean(c_img, x_valid)

        pos_img = self.fetch_pos(x.shape[1], x.device)
        pos_txt = None
        joint_mask = self.build_joint_attention_mask(text_valid, x_valid)

        for block in self.blocks:
            x, conds = block(
                x,
                conds,
                c_img,
                c_txt,
                pos_img,
                pos_txt,
                joint_mask,
            )
        return self.final_layer(x)

    def forward_loss(self, latents, y, m_lens):
        target = self._canonicalize_raw_motion(latents)
        frame_count = target.shape[2]
        non_pad_mask = lengths_to_mask(m_lens, frame_count)
        target = torch.where(
            non_pad_mask[:, None, :, None],
            target,
            torch.zeros_like(target),
        )

        cond_vector, conds_mask = self.prepare_text_condition(y)

        attention_mask = (
            non_pad_mask.unsqueeze(-1)
            .expand(-1, -1, self.patches_per_frame)
            .reshape(non_pad_mask.shape[0], -1)
        )
        model_kwargs = dict(
            conds=cond_vector,
            attention_mask=attention_mask,
            conds_mask=conds_mask,
        )
        loss_dict = self._flow_training_losses(target, model_kwargs, dim=(1, 3))
        loss = loss_dict["loss"]
        return (loss * non_pad_mask).sum() / non_pad_mask.sum()

    def generate_full_sequence(
        self,
        conds,
        m_lens,
        cond_scale=1.0,
    ):
        device = next(self.parameters()).device
        m_lens = torch.as_tensor(m_lens, device=device, dtype=torch.long)
        frame_count = int(m_lens.max().item())
        batch_size = len(m_lens)
        noise_scale = self.noise_scale
        sampler = self.sampler
        sampling_timesteps = self.sampling_timesteps
        if sampling_timesteps <= 1:
            raise ValueError(
                "sampling_timesteps must be greater than 1, "
                f"got {sampling_timesteps}."
            )

        cond_vector, conds_mask = self.prepare_text_condition(conds)
        valid_mask = lengths_to_mask(m_lens, frame_count)
        noise = (
            torch.randn(
                batch_size,
                self.input_dim,
                frame_count,
                self.raw_joint_count,
                device=device,
            )
            * noise_scale
        )
        attention_mask = (
            valid_mask.unsqueeze(-1)
            .expand(-1, -1, self.patches_per_frame)
            .reshape(batch_size, -1)
        )
        if cond_scale != 1.0:
            cond_vector = torch.cat([cond_vector, torch.zeros_like(cond_vector)], dim=1)
            conds_mask = torch.cat([conds_mask, conds_mask], dim=0)
            attention_mask = attention_mask.repeat(2, 1)
            noise = torch.cat([noise, noise], dim=0)

        model_kwargs = dict(
            conds=cond_vector,
            conds_mask=conds_mask,
            attention_mask=attention_mask,
            cfg=cond_scale,
        )
        model_fn = self.gen_diffusion.sample_ode(
            sampling_method=sampler,
            num_steps=sampling_timesteps,
        )
        sampled = model_fn(noise, self.forward_with_CFG, **model_kwargs)[-1]
        if cond_scale != 1.0:
            sampled, _ = sampled.chunk(2, dim=0)
        sampled = torch.where(
            valid_mask[:, None, :, None], sampled, torch.zeros_like(sampled)
        )
        return (
            sampled.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch_size, frame_count, 1, -1)
        )


def _build_mmdit_xyz(layer: int, **kwargs) -> MMDiT_xyz:
    defaults = {
        "latent_dim": layer * 64,
        "ff_size": layer * 64 * 2,
        "num_layers": layer,
        "num_heads": layer // 2,
        "dropout": 0,
        "clip_dim": 768,
        "cond_mask_prob": 0.1,
    }
    defaults.update(kwargs)
    return MMDiT_xyz(**defaults)


def build_model(**kwargs):
    return _build_mmdit_xyz(8, **kwargs)
