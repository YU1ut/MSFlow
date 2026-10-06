import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from jit_diffusions.transport import Sampler, create_transport
from jit_diffusions.transport import path as transport_path
from jit_diffusions.transport.utils import mean_flat
from models.BERT.BERT_encoder import load_bert
from models.modules import MMDiTBlockT2I, rope_params
from models.token_refiner import SingleTokenRefiner


class MMDiT(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0,
        clip_dim=512,
        cond_mask_prob=0.1,
        num_frame_per_block=1,
        is_causal=True,
        flow_output_type="velocity",
        flow_t_sampler="uniform",
        sigma_min=0.05,
        max_length=49,
        use_single_token_refiner=True,
        bert_model_path="distilbert/distilbert-base-uncased",
        noise_scale=5.0,
        sampler="euler",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_frame_per_block = int(num_frame_per_block)
        if self.num_frame_per_block <= 0:
            raise ValueError(
                "num_frame_per_block must be positive, "
                f"got {self.num_frame_per_block}."
            )

        self.cond_mask_prob = cond_mask_prob
        self.is_causal = is_causal
        self.bert_model_path = bert_model_path
        if self.is_causal:
            print("Using Causal Transformer")
        else:
            print("Using Non-Causal Transformer")

        self.flow_output_type = flow_output_type
        self.flow_t_sampler = self._normalize_flow_t_sampler(flow_t_sampler)
        self.flow_t_logit_mu = -0.8
        self.sigma_min = sigma_min
        self.sampler = str(sampler or "euler")

        self.max_length = max_length

        self.sampling_timesteps = 50
        if self.sampling_timesteps <= 1:
            raise ValueError(
                "sampling_timesteps must be greater than 1, "
                f"got {self.sampling_timesteps}."
            )
        self.noise_scale = float(noise_scale)
        if self.noise_scale < 0.0:
            raise ValueError(
                f"noise_scale must be non-negative, got {self.noise_scale}."
            )

        print("Loading MMDiT...")
        self.t_embedder = TimestepEmbedder(self.latent_dim)
        self.x_embedder = nn.Linear(self.input_dim, self.latent_dim)

        self.y_embedder = nn.Linear(self.clip_dim, self.latent_dim)
        self.use_single_token_refiner = bool(use_single_token_refiner)
        self.y_token_refiner = None
        if self.use_single_token_refiner:
            self.y_token_refiner = SingleTokenRefiner(
                input_dim=self.latent_dim,
                feat_dim=self.latent_dim,
                num_heads=self.num_heads,
                num_layers=2,
                dropout=self.dropout,
            )

        self.final_layer = nn.Linear(self.latent_dim, self.input_dim)
        head_dim = self.latent_dim // self.num_heads
        self.freqs = rope_params(self.max_length, head_dim)

        print("DiT causal init")
        print("is_causal: ", self.is_causal)
        print("latent_dim: ", self.latent_dim)
        print("ff_size: ", self.ff_size)
        print("num_heads: ", self.num_heads)
        print("num_layers: ", self.num_layers)
        print("num_frame_per_block: ", self.num_frame_per_block)
        print("max_length: ", self.max_length)
        print("Using MMDiT")
        print("use_single_token_refiner: ", self.use_single_token_refiner)
        print("noise_scale: ", self.noise_scale)
        print("sampler: ", self.sampler)
        print("MMDiTBlockT2I layers: ", self.num_layers)

        self.blocks = nn.ModuleList(
            [
                MMDiTBlockT2I(
                    self.latent_dim,
                    self.num_heads,
                    mlp_ratio=self.ff_size / self.latent_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.global_y_embedder = nn.Linear(self.latent_dim, self.latent_dim)
        self.initialize_weights()

        print("Loading BERT text encoder...")
        self.clip_model = self.load_and_freeze_text_encoder()

        print(
            "Using JiT Flow diffusion: "
            f"output_type={self.flow_output_type}, "
            f"t_sampler={self.flow_t_sampler}, "
            f"logit_mu={self.flow_t_logit_mu}, "
            f"sigma_min={self.sigma_min}, "
            f"step_sampler={self.sampler}"
        )
        self.train_diffusion = create_transport(
            prediction_output=self.flow_output_type,
            path_type="Linear",
            x_t_sampler=self.flow_t_sampler,
            logit_mu=self.flow_t_logit_mu,
            sigma_min=self.sigma_min,
            num_frame_per_block=self.num_frame_per_block,
            is_causal=False,
            t_per_frame=False,
        )
        self.gen_diffusion = Sampler(self.train_diffusion)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.t_embedder.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        self.x_embedder.apply(_basic_init)
        self.y_embedder.apply(_basic_init)
        if self.y_token_refiner is not None:
            self.y_token_refiner.apply(_basic_init)
            if (
                self.y_token_refiner.input_embedder.weight.shape[0]
                == self.y_token_refiner.input_embedder.weight.shape[1]
            ):
                nn.init.eye_(self.y_token_refiner.input_embedder.weight)
                nn.init.constant_(self.y_token_refiner.input_embedder.bias, 0)
            for block in self.y_token_refiner.individual_token_refiner.blocks:
                nn.init.constant_(block.adaLN_modulation.linear.weight, 0)
                nn.init.constant_(block.adaLN_modulation.linear.bias, 0)
        self.final_layer.apply(_basic_init)
        self.blocks.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation_img[0].weight, 0)
            nn.init.constant_(block.adaLN_modulation_img[0].bias, 0)
            nn.init.constant_(block.adaLN_modulation_txt[0].weight, 0)
            nn.init.constant_(block.adaLN_modulation_txt[0].bias, 0)

        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def load_and_freeze_text_encoder(self):
        return load_bert(self.bert_model_path)

    def mask_cond(self, cond):
        _, batch_size, _ = cond.shape
        if self.training and self.cond_mask_prob > 0.0:
            keep_mask = 1.0 - torch.bernoulli(
                torch.ones(batch_size, device=cond.device, dtype=cond.dtype)
                * self.cond_mask_prob
            ).view(1, batch_size, 1)
        else:
            keep_mask = torch.ones(
                1, batch_size, 1, device=cond.device, dtype=cond.dtype
            )

        return cond * keep_mask

    def encode_text(self, raw_text):
        enc_text, mask = self.clip_model(raw_text)
        enc_text = enc_text.permute(1, 0, 2)
        return enc_text, mask

    def prepare_text_condition(self, condition):
        with torch.no_grad():
            return self.encode_text(condition)

    def fetch_pos(self, length, device):
        if self.freqs.shape[0] < length:
            self.freqs = rope_params(
                length,
                self.latent_dim // self.num_heads,
            )
        return self.freqs[:length].to(device)

    def normalize_valid_mask(self, mask, batch_size, length, device, name):
        if mask is None:
            return torch.ones(batch_size, length, device=device, dtype=torch.bool)

        mask = mask.to(device=device)
        while mask.dim() > 2 and mask.size(1) == 1:
            mask = mask.squeeze(1)
        if mask.dim() != 2:
            raise ValueError(
                f"{name} must be broadcastable to shape [B, L], "
                f"got {tuple(mask.shape)}."
            )
        if mask.shape[0] != batch_size:
            raise ValueError(f"{name} batch mismatch: {mask.shape[0]} vs {batch_size}.")

        mask = mask.bool()
        if mask.shape[1] > length:
            return mask[:, :length]
        if mask.shape[1] < length:
            pad = torch.zeros(
                batch_size,
                length - mask.shape[1],
                device=device,
                dtype=torch.bool,
            )
            return torch.cat([mask, pad], dim=1)
        return mask

    def masked_mean(self, values, mask):
        if values.shape[1] == 1:
            return values
        if values.shape[1] != mask.shape[1]:
            raise ValueError(
                "Condition and mask lengths must match unless condition length is 1, "
                f"got {values.shape[1]} and {mask.shape[1]}."
            )
        weights = mask.to(dtype=values.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * weights).sum(dim=1, keepdim=True) / denom

    def single_token_refiner_timestep(self, t, x_valid):
        batch_size = x_valid.shape[0]
        if not torch.is_tensor(t):
            return torch.full(
                (batch_size,),
                float(t),
                device=x_valid.device,
                dtype=torch.float32,
            )

        t = t.to(device=x_valid.device)
        if t.dim() == 0:
            return t.float().expand(batch_size)
        if t.dim() == 1:
            if t.shape[0] == 1 and batch_size != 1:
                return t.float().expand(batch_size)
            if t.shape[0] != batch_size:
                raise ValueError(
                    "SingleTokenRefiner timestep batch mismatch: "
                    f"{t.shape[0]} vs {batch_size}."
                )
            return t.float()

        if t.shape[0] == 1 and batch_size != 1:
            t = t.expand(batch_size, *t.shape[1:])
        elif t.shape[0] != batch_size:
            raise ValueError(
                "SingleTokenRefiner timestep batch mismatch: "
                f"{t.shape[0]} vs {batch_size}."
            )

        t = t.reshape(batch_size, -1).float()
        if t.shape[1] == x_valid.shape[1]:
            weights = x_valid.to(dtype=t.dtype)
            denom = weights.sum(dim=1).clamp_min(1.0)
            return (t * weights).sum(dim=1) / denom
        return t.mean(dim=1)

    @staticmethod
    def _normalize_flow_t_sampler(flow_t_sampler):
        normalized = str(flow_t_sampler).strip().lower()
        if normalized not in {"logit", "uniform"}:
            raise ValueError(
                "flow_t_sampler must be logit or uniform; " f"got {flow_t_sampler!r}."
            )
        return normalized

    def build_causal_allowed(self, length, device):
        ends = torch.zeros(length, device=device, dtype=torch.long)
        for start in range(0, length, self.num_frame_per_block):
            ends[start : start + self.num_frame_per_block] = min(
                start + self.num_frame_per_block,
                length,
            )

        q_idx = torch.arange(length, device=device).unsqueeze(1)
        k_idx = torch.arange(length, device=device).unsqueeze(0)
        return (k_idx < ends[q_idx]) | (q_idx == k_idx)

    def build_joint_attention_mask(self, text_valid, x_valid):
        batch_size, text_len = text_valid.shape
        x_len = x_valid.shape[1]
        total_len = text_len + x_len
        key_valid = torch.cat([text_valid, x_valid], dim=1)
        allowed = key_valid[:, None, None, :].expand(
            batch_size,
            1,
            total_len,
            total_len,
        )
        if not self.is_causal:
            return allowed

        structure = torch.zeros(
            total_len,
            total_len,
            device=x_valid.device,
            dtype=torch.bool,
        )
        structure[:text_len, :text_len] = True
        structure[text_len:, :text_len] = True
        structure[text_len:, text_len:] = self.build_causal_allowed(
            x_len, x_valid.device
        )
        return allowed & structure.view(1, 1, total_len, total_len)

    def masked_text_mean(self, values, mask):
        if values.shape[1] != mask.shape[1]:
            raise ValueError(
                "Text condition and mask lengths must match, "
                f"got {values.shape[1]} and {mask.shape[1]}."
            )

        weights = mask.to(dtype=values.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * weights).sum(dim=1, keepdim=True) / denom

    def build_global_conds(self, conds, text_valid, text_global_embedder, dtype):
        pooled_conds = self.masked_text_mean(conds, text_valid).to(dtype=dtype)
        return text_global_embedder(pooled_conds)

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
        x = self.x_embedder(x)
        x = x.flatten(2)
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

        x = rearrange(x, "batch len dim -> batch len 1 dim")
        return self.final_layer(x)

    def _sample_flow_training_state(self, target):
        return self.train_diffusion.sample(target)

    def _flow_training_losses(self, target, model_kwargs, dim=(2, 3)):
        transport = self.train_diffusion
        t, x0, x1 = self._sample_flow_training_state(target)

        noise_scale = self.noise_scale
        if noise_scale != 1.0:
            x0 = x0 * noise_scale

        t, xt, ut = transport.path_sampler.plan(t, x0, x1)
        model_output = self.forward_step(xt, t, **model_kwargs)
        batch_size, *_, channels = xt.shape
        expected_size = (batch_size, *xt.size()[1:-1], channels)
        if model_output.size() != expected_size:
            raise ValueError(
                f"Expected model output size {expected_size}, "
                f"got {tuple(model_output.size())}."
            )

        terms = {"pred": model_output, "xt": xt, "t": t}
        pred_velocity = transport._convert_output_to_velocity(xt, t, model_output)
        if transport.prediction_output == "x":
            sigma_t, _ = transport.path_sampler.compute_sigma_t(
                transport_path.expand_t_like_x(t, xt)
            )
            denom = torch.clamp(sigma_t, min=transport.sigma_min)
            velocity_target = (x1 - xt) / denom
        else:
            velocity_target = ut
        loss = mean_flat((pred_velocity - velocity_target) ** 2, dim=dim)
        terms["loss"] = loss
        return terms

    def forward_loss(self, latents, y, m_lens):
        _, sequence_length, _, _ = latents.shape

        non_pad_mask = lengths_to_mask(m_lens, sequence_length)
        latents = torch.where(
            non_pad_mask.unsqueeze(-1).unsqueeze(-1), latents, torch.zeros_like(latents)
        )

        target = latents.clone()

        cond_vector, conds_mask = self.prepare_text_condition(y)

        attention_mask = non_pad_mask

        model_kwargs = dict(
            conds=cond_vector,
            attention_mask=attention_mask,
            conds_mask=conds_mask,
        )
        loss_dict = self._flow_training_losses(target, model_kwargs, dim=(2, 3))
        loss = loss_dict["loss"]
        loss = (loss * non_pad_mask).sum() / non_pad_mask.sum()

        return loss

    def forward_with_CFG(
        self,
        x,
        t,
        conds,
        conds_mask,
        attention_mask,
        cfg=1.0,
    ):
        if not cfg == 1.0:
            half = x[: len(x) // 2]
            x = torch.cat([half, half], dim=0)
        x = self.forward_step(
            x,
            t,
            conds,
            conds_mask,
            attention_mask,
        )
        if not cfg == 1.0:
            cond_eps, uncond_eps = torch.split(x, len(x) // 2, dim=0)
            half_eps = uncond_eps + cfg * (cond_eps - uncond_eps)
            x = torch.cat([half_eps, half_eps], dim=0)
        return x

    def generate_full_sequence(
        self,
        conds,
        m_lens,
        cond_scale=1.0,
    ):
        device = next(self.parameters()).device
        if torch.is_tensor(m_lens):
            m_lens = m_lens.to(device=device, dtype=torch.long)
        else:
            m_lens = torch.tensor(m_lens, device=device, dtype=torch.long)
        sequence_length = int(m_lens.max().item())
        batch_size = len(m_lens)
        noise_scale = self.noise_scale
        sampler = self.sampler
        sampling_timesteps = self.sampling_timesteps
        if sampling_timesteps <= 1:
            raise ValueError(
                "sampling_timesteps must be greater than 1, "
                f"got {sampling_timesteps}."
            )

        with torch.no_grad():
            cond_vector, conds_mask = self.encode_text(conds)

        padding_mask = ~lengths_to_mask(m_lens, sequence_length)

        noise = (
            torch.randn(batch_size, sequence_length, 1, self.input_dim, device=device)
            * noise_scale
        )
        if not cond_scale == 1.0:
            cond_vector = torch.cat([cond_vector, torch.zeros_like(cond_vector)], dim=1)
            conds_mask = torch.cat([conds_mask, conds_mask], dim=0)
            noise = torch.cat([noise, noise], dim=0)

        attention_mask = ~padding_mask
        model_kwargs = dict(
            conds=cond_vector,
            conds_mask=conds_mask,
            attention_mask=attention_mask,
            cfg=cond_scale,
        )
        sample_fn = self.forward_with_CFG

        if not cond_scale == 1:
            model_kwargs["attention_mask"] = attention_mask.repeat(2, 1)

        model_fn = self.gen_diffusion.sample_ode(
            sampling_method=sampler,
            num_steps=sampling_timesteps,
        )  # default to ode sampling
        sampled_token_latent = model_fn(noise, sample_fn, **model_kwargs)[-1]

        if not cond_scale == 1:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
        return sampled_token_latent

    def generate(self, conds, m_lens, cond_scale=1.0):
        return self.generate_full_sequence(
            conds,
            m_lens,
            cond_scale=cond_scale,
        )


def build_model(**kwargs):
    layer = 8
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
    return MMDiT(**defaults)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=torch.float32):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        if t.dim() == 1:
            t = t.unsqueeze(1)

        B, L = t.shape
        t = t.reshape(B * L)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
        ).to(device=t.device, dtype=dtype)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        embedding = embedding.reshape(B, L, -1)
        return embedding

    def forward(self, t, dtype=torch.bfloat16):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask  # (b, len)
