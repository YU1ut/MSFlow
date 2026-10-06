from typing import Optional, Tuple

import torch
import torch.nn as nn


def apply_adaln(x, shift, scale):
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MMDiTJointAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim should be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)

        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pos_img: torch.Tensor,
        pos_txt: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_img_tokens, channels = x.shape
        batch_text, num_txt_tokens, text_channels = y.shape
        if batch_size != batch_text or channels != text_channels:
            raise ValueError("x and y must share batch and channel dims")

        qkv_x = self.qkv_x(x).reshape(
            batch_size,
            num_img_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv_x = qkv_x.permute(2, 0, 1, 3, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        qkv_y = self.qkv_y(y).reshape(
            batch_size,
            num_txt_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv_y = qkv_y.permute(2, 0, 1, 3, 4)
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None and num_txt_tokens > 0:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        qx = qx.transpose(1, 2)
        kx = kx.transpose(1, 2)
        vx = vx.transpose(1, 2)
        qy = qy.transpose(1, 2)
        ky = ky.transpose(1, 2)
        vy = vy.transpose(1, 2)

        q_joint = torch.cat([qy, qx], dim=2)
        k_joint = torch.cat([ky, kx], dim=2)
        v_joint = torch.cat([vy, vx], dim=2)

        out_joint = torch.nn.functional.scaled_dot_product_attention(
            q_joint,
            k_joint,
            v_joint,
            dropout_p=0.0,
            attn_mask=attn_mask,
        )
        out_y = out_joint[:, :, :num_txt_tokens, :]
        out_x = out_joint[:, :, num_txt_tokens:, :]

        out_y = out_y.transpose(1, 2).reshape(batch_size, num_txt_tokens, channels)
        out_x = out_x.transpose(1, 2).reshape(batch_size, num_img_tokens, channels)

        out_x = self.proj_drop_x(self.proj_x(out_x))
        out_y = self.proj_drop_y(self.proj_y(out_y))
        return out_x, out_y


class MMDiTBlockT2I(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        groups: int,
        mlp_ratio: float = 4.0,
        adaLN_modulation_img=None,
        adaLN_modulation_txt=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups
        self.head_dim = hidden_size // groups

        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation_img = adaLN_modulation_img or nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation_txt = adaLN_modulation_txt or nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x,
        y,
        c_img,
        c_txt,
        pos_img,
        pos_txt=None,
        attn_mask=None,
    ):
        img_chunks = self.adaLN_modulation_img(c_img).chunk(6, dim=-1)
        txt_chunks = self.adaLN_modulation_txt(c_txt).chunk(6, dim=-1)
        (
            shift_msa_x,
            scale_msa_x,
            gate_msa_x,
            shift_mlp_x,
            scale_mlp_x,
            gate_mlp_x,
        ) = img_chunks
        (
            shift_msa_y,
            scale_msa_y,
            gate_msa_y,
            shift_mlp_y,
            scale_mlp_y,
            gate_mlp_y,
        ) = txt_chunks

        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y

        x = x + gate_mlp_x * self.mlp_x(
            apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x)
        )
        y = y + gate_mlp_y * self.mlp_y(
            apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y)
        )
        return x, y


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    # compute in float32 for speed; returns complex64
    inv = torch.arange(0, dim, 2, dtype=torch.float32) / float(dim)
    base = torch.tensor(float(theta), dtype=torch.float32)
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float32), 1.0 / torch.pow(base, inv)
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs
