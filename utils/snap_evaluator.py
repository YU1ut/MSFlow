"""SnapMoGen text-motion evaluator adapted from the CVPR sample code."""

from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from omegaconf import OmegaConf
from transformers import AutoTokenizer, T5EncoderModel


def _length_to_mask(lengths, max_length, device):
    lengths = lengths.to(device)
    return torch.arange(max_length, device=device).expand(
        len(lengths), max_length
    ) < lengths.unsqueeze(1)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=1)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class _PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoding = torch.zeros(max_length, dim)
        position = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs):
        return self.dropout(inputs + self.pe[:, : inputs.shape[1]])


class _Encoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        variational,
        latent_dim,
        feedforward_dim,
        num_layers,
        num_heads,
        dropout,
        activation,
    ):
        super().__init__()
        self.projection = nn.Linear(feature_dim, latent_dim)
        self.vae = variational
        self.nbtokens = 2 if variational else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))
        self.sequence_pos_encoding = _PositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.apply(_init_weights)

    def encode(self, inputs, mask, sample_mean=True):
        inputs = self.projection(inputs)
        tokens = repeat(self.tokens, "n d -> b n d", b=len(inputs))
        sequence = self.sequence_pos_encoding(torch.cat((tokens, inputs), dim=1))
        token_mask = torch.ones(
            (len(inputs), self.nbtokens), dtype=torch.bool, device=inputs.device
        )
        mask = torch.cat((token_mask, mask), dim=1)
        output = self.seqTransEncoder(sequence, src_key_padding_mask=~mask)
        fid_embedding = output[:, 0]
        distribution = self.linear(output[:, : self.nbtokens]).unbind(1)
        if not self.vae:
            return fid_embedding, distribution[0], None

        mean, log_variance = distribution
        log_variance = torch.clamp(log_variance, -10.0, 10.0)
        if sample_mean:
            embedding = mean
        else:
            deviation = log_variance.mul(0.5).exp()
            embedding = mean + torch.randn_like(deviation) * deviation
        return fid_embedding, embedding, (mean, log_variance)


class _T5TextEncoder:
    def __init__(self, model_path, max_length, device, local_files_only):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            legacy=False,
        )
        self.model = T5EncoderModel.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        ).eval()
        self.model.requires_grad_(False)
        self.model.to(device)

    @torch.no_grad()
    def encode(self, texts):
        tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        embeddings = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return embeddings, attention_mask.bool()


class SnapMoGenEvaluator:
    """Load the official SnapMoGen evaluator and expose its embedding API."""

    def __init__(
        self,
        config_path,
        checkpoint_path,
        device,
        text_model_path=None,
        local_files_only=False,
    ):
        config = OmegaConf.load(config_path)
        encoder_config = config.latent_encoder
        text_config = config.text_encoder
        self.motion_feature_dim = int(config.data.dim_pose)

        self.motion_encoder = _Encoder(
            self.motion_feature_dim,
            bool(config.vae),
            int(encoder_config.latent_dim),
            int(encoder_config.ff_size),
            int(encoder_config.num_layers),
            int(encoder_config.num_heads),
            float(encoder_config.dropout),
            str(encoder_config.activation),
        )
        self.text_encoder = _Encoder(
            int(text_config.nfeats),
            bool(config.vae),
            int(text_config.latent_dim),
            int(text_config.ff_size),
            int(text_config.num_layers),
            int(text_config.num_heads),
            float(text_config.dropout),
            str(text_config.activation),
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.motion_encoder.load_state_dict(checkpoint["latent_enc"])
        self.text_encoder.load_state_dict(checkpoint["text_enc"])
        self.motion_encoder.to(device).eval()
        self.text_encoder.to(device).eval()
        self.motion_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        resolved_text_model_path = text_model_path or str(config.text_embedder.version)
        self.text_embedder = _T5TextEncoder(
            resolved_text_model_path,
            int(config.data.max_text_length),
            device,
            local_files_only,
        )
        self.device = device
        print(
            "Loaded SnapMoGen evaluator from "
            f"{pjoin(str(config.exp.name), 'model')} at epoch {checkpoint['ep']}"
        )

    @torch.no_grad()
    def encode_text(self, texts):
        embeddings, mask = self.text_embedder.encode(texts)
        _, vectors, _ = self.text_encoder.encode(embeddings, mask, sample_mean=True)
        return vectors

    @torch.no_grad()
    def encode_motion(self, motions, lengths):
        motions = motions[..., : self.motion_feature_dim]
        mask = _length_to_mask(lengths, motions.shape[1], motions.device)
        return self.motion_encoder.encode(motions, mask, sample_mean=True)[:2]
