from __future__ import annotations

import argparse
from os.path import join as pjoin
from typing import Any

import torch

from utils.config_utils import load_yaml_config
from utils.mmdit_model_utils import resolve_mmdit_factory

MODEL_DEFAULTS: dict[str, Any] = {
    "max_length": 1024,
    "is_causal": False,
    "num_frame_per_block": 1,
    "use_single_token_refiner": True,
    "bert_model_path": "distilbert/distilbert-base-uncased",
    "noise_scale": 5.0,
    "flow_output_type": "x",
    "flow_t_sampler": "uniform",
    "sigma_min": 0.05,
    "sampler": "euler",
}


def _resolve_value(
    args: argparse.Namespace,
    saved_cfg: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    return saved_cfg.get(key, default)


def _resolve_input_dim(
    args: argparse.Namespace,
    saved_cfg: dict[str, Any],
) -> int:
    raw_joint_count = int(_resolve_value(args, saved_cfg, "raw_joint_count", 22))
    raw_joint_dim = int(_resolve_value(args, saved_cfg, "raw_joint_dim", 3))
    if (raw_joint_count, raw_joint_dim) != (22, 3):
        raise ValueError(
            "Raw MMDiT inference expects 22 XYZ joints, got "
            f"{raw_joint_count}x{raw_joint_dim}."
        )

    explicit = _resolve_value(args, saved_cfg, "input_dim", None)
    if explicit is None:
        explicit = _resolve_value(args, saved_cfg, "model_input_dim", None)
    input_dim = raw_joint_count * raw_joint_dim
    if explicit is not None and int(explicit) != input_dim:
        raise ValueError(f"Raw model input_dim must be {input_dim}, got {explicit}.")
    return input_dim


def _build_model_kwargs(
    args: argparse.Namespace,
    saved_cfg: dict[str, Any],
    input_dim: int,
) -> dict[str, Any]:
    kwargs = {
        key: _resolve_value(args, saved_cfg, key, default)
        for key, default in MODEL_DEFAULTS.items()
    }
    kwargs.update(
        {
            "input_dim": input_dim,
            "raw_joint_count": 22,
            "raw_joint_dim": 3,
            "patch_size": tuple(saved_cfg.get("patch_size", (1, 22))),
            "stride_size": tuple(saved_cfg.get("stride_size", (1, 22))),
        }
    )
    return kwargs


def _validate_checkpoint_keys(
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    incompatible = [key for key in missing_keys if not key.startswith("clip_model.")]
    if unexpected_keys or incompatible:
        details = []
        if unexpected_keys:
            details.append("unexpected keys: " + ", ".join(unexpected_keys[:20]))
        if incompatible:
            details.append("missing keys: " + ", ".join(incompatible[:20]))
        raise RuntimeError("Could not load checkpoint: " + "; ".join(details))


def load_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    run_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    saved_cfg_path = pjoin(run_dir, "config.yaml")
    saved_cfg = load_yaml_config(saved_cfg_path)
    if not saved_cfg:
        raise FileNotFoundError(f"Saved model config not found: {saved_cfg_path}")

    input_dim = _resolve_input_dim(args, saved_cfg)
    model_kwargs = _build_model_kwargs(args, saved_cfg, input_dim)
    if int(args.n_frames) > int(model_kwargs["max_length"]):
        raise ValueError(
            f"Requested n_frames={args.n_frames} exceeds model max_length="
            f"{model_kwargs['max_length']}."
        )

    requested_variant = _resolve_value(
        args,
        saved_cfg,
        "model_variant",
        "MMDiT_xyz",
    )
    factory, resolved_variant = resolve_mmdit_factory(requested_variant)
    print(f"Loading {resolved_variant} with input_dim={input_dim}")
    model = factory(**model_kwargs)

    checkpoint_name = args.ckpt
    if not checkpoint_name.endswith(".tar"):
        checkpoint_name += ".tar"
    checkpoint_path = pjoin(run_dir, "model", checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if args.checkpoint_key not in checkpoint:
        raise KeyError(
            f"Checkpoint key '{args.checkpoint_key}' not found in {checkpoint_path}."
        )
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint[args.checkpoint_key],
        strict=False,
    )
    _validate_checkpoint_keys(missing_keys, unexpected_keys)
    model.eval().to(device)
    return model, input_dim
