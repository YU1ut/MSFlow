import csv
import os
import random
from os.path import join as pjoin
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from utils.config_utils import load_yaml_config
from utils.mmdit_model_utils import resolve_mmdit_factory
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.raw_joint_utils import load_raw_joint_mean_std, unflatten_raw_joints_np

MODEL_DEFAULTS: dict[str, Any] = {
    "max_length": 1024,
    "is_causal": True,
    "num_frame_per_block": 1,
    "use_single_token_refiner": True,
    "bert_model_path": "distilbert/distilbert-base-uncased",
    "noise_scale": 5.0,
    "flow_output_type": "x",
    "flow_t_sampler": "uniform",
    "sigma_min": 1e-5,
    "sampler": "myheun2",
}


def load_prompts_and_lengths(cfg: DictConfig) -> tuple[list[str], list[int]]:
    prompt_csv = cfg.get("prompt_csv")
    if prompt_csv:
        csv_path = Path(prompt_csv).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"Prompt CSV not found: {csv_path}")
        prompts: list[str] = []
        lengths: list[int] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row_idx, row in enumerate(reader, start=2):
                prompt = (row.get("prompt") or "").strip()
                length_str = (row.get("length") or "").strip()
                if not prompt:
                    raise ValueError(
                        f"Missing prompt at {csv_path}:{row_idx}. "
                        "Expected columns: prompt,length"
                    )
                if not length_str:
                    raise ValueError(
                        f"Missing length at {csv_path}:{row_idx}. "
                        "Expected columns: prompt,length"
                    )
                try:
                    length = int(length_str)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid length at {csv_path}:{row_idx}: {length_str}"
                    ) from exc
                if length <= 0:
                    raise ValueError(
                        f"Length must be > 0 at {csv_path}:{row_idx}, got {length}"
                    )
                prompts.append(prompt)
                lengths.append(length)

        if not prompts:
            raise ValueError(f"No prompts found in CSV: {csv_path}")

        print(f"Loaded {len(prompts)} prompts from: {csv_path}")
        return prompts, lengths

    if isinstance(cfg.input_text, str):
        return [cfg.input_text] * cfg.num_samples, [cfg.length] * cfg.num_samples

    prompts = [str(prompt) for prompt in cfg.input_text]
    lengths = [int(length) for length in cfg.length]
    if len(prompts) != len(lengths):
        raise ValueError(
            "input_text and length lists must contain the same number of items."
        )
    if not prompts:
        raise ValueError("input_text and length lists must not be empty.")
    return prompts, lengths


def _resolve_checkpoint_file(model_dir: str, checkpoint_name: str) -> str:
    checkpoint_file = (
        checkpoint_name
        if checkpoint_name.endswith(".tar")
        else f"{checkpoint_name}.tar"
    )
    return pjoin(model_dir, checkpoint_file)


def _resolve_value(
    cfg: DictConfig, saved_cfg: dict[str, Any], key: str, default: Any
) -> Any:
    cfg_val = cfg.get(key)
    if cfg_val is not None:
        return cfg_val
    if key in saved_cfg:
        return saved_cfg[key]
    return default


def _resolve_raw_joint_shape(
    cfg: DictConfig, saved_cfg: dict[str, Any]
) -> tuple[int, int] | None:
    raw_joint_count_value = _resolve_value(cfg, saved_cfg, "raw_joint_count", None)
    if raw_joint_count_value is None:
        return None

    raw_joint_count = int(raw_joint_count_value)
    raw_joint_dim = int(_resolve_value(cfg, saved_cfg, "raw_joint_dim", 3))
    if raw_joint_count != 22 or raw_joint_dim != 3:
        raise ValueError(
            "MMDiT_xyz demo expects 22 XYZ joints, got "
            f"raw_joint_count={raw_joint_count} and "
            f"raw_joint_dim={raw_joint_dim}."
        )
    return raw_joint_count, raw_joint_dim


def _resolve_input_dim(cfg: DictConfig, saved_cfg: dict[str, Any]) -> int:
    raw_joint_shape = _resolve_raw_joint_shape(cfg, saved_cfg)
    if raw_joint_shape is not None:
        return raw_joint_shape[0] * raw_joint_shape[1]

    for key in ("input_dim", "model_input_dim"):
        value = _resolve_value(cfg, saved_cfg, key, None)
        if value is not None:
            return int(value)

    run_name = str(_resolve_value(cfg, saved_cfg, "name", ""))
    if run_name.endswith("_67"):
        return 67
    return 263


def _build_model_kwargs(
    cfg: DictConfig, saved_cfg: dict[str, Any], input_dim: int
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "input_dim": input_dim,
    }
    for key, default in MODEL_DEFAULTS.items():
        value = _resolve_value(cfg, saved_cfg, key, default)
        if value is not None:
            model_kwargs[key] = value

    raw_joint_shape = _resolve_raw_joint_shape(cfg, saved_cfg)
    if raw_joint_shape is not None:
        raw_joint_count, raw_joint_dim = raw_joint_shape
        model_kwargs.update(
            {
                "raw_joint_dim": raw_joint_dim,
                "raw_joint_count": raw_joint_count,
                "patch_size": tuple(
                    _resolve_value(cfg, saved_cfg, "patch_size", (1, 22))
                ),
                "stride_size": tuple(
                    _resolve_value(cfg, saved_cfg, "stride_size", (1, 22))
                ),
            }
        )
    return model_kwargs


def _validate_checkpoint_keys(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    allow_unexpected: bool = False,
) -> None:
    if not allow_unexpected and unexpected_keys:
        raise RuntimeError(
            "Unexpected keys in checkpoint state_dict: " + ", ".join(unexpected_keys)
        )

    incompatible = [k for k in missing_keys if not k.startswith(("clip_model.",))]
    if incompatible:
        raise RuntimeError(
            "Missing non-CLIP keys in checkpoint state_dict: " + ", ".join(incompatible)
        )


def run(cfg: DictConfig) -> None:
    torch.backends.cudnn.benchmark = False
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.dataset_name != "t2m":
        raise NotImplementedError(f"Unsupported dataset: {cfg.dataset_name}")
    data_root = f"{cfg.dataset_dir}/HumanML3D/"

    run_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name)
    model_dir = pjoin(run_dir, "model")
    checkpoint_path = _resolve_checkpoint_file(model_dir, cfg.checkpoint)

    saved_cfg_path = pjoin(run_dir, "config.yaml")
    saved_cfg = load_yaml_config(saved_cfg_path)
    if saved_cfg:
        print(f"Loaded saved model config from {saved_cfg_path}")
    else:
        print(
            f"No saved model config found at {saved_cfg_path}. "
            "Using script defaults."
        )

    raw_joint_shape = _resolve_raw_joint_shape(cfg, saved_cfg)
    if raw_joint_shape is None:
        mean = np.load(pjoin(data_root, "Mean.npy"))
        std = np.load(pjoin(data_root, "Std.npy"))
        motion_representation = "HumanML3D features"
    else:
        raw_joint_count, raw_joint_dim = raw_joint_shape
        raw_normalization = _resolve_value(
            cfg,
            saved_cfg,
            "raw_normalization",
            "shared_xyz",
        )
        mean, std = load_raw_joint_mean_std(
            cfg.dataset_name,
            data_root=data_root,
            mean_path=_resolve_value(cfg, saved_cfg, "raw_mean_path", None),
            std_path=_resolve_value(cfg, saved_cfg, "raw_std_path", None),
            normalization=raw_normalization,
            joint_count=raw_joint_count,
            joint_dim=raw_joint_dim,
            split_file=pjoin(data_root, "train.txt"),
        )
        motion_representation = f"raw XYZ joints ({raw_normalization} normalization)"

    dim_pose = _resolve_input_dim(cfg, saved_cfg)
    model_kwargs = _build_model_kwargs(cfg, saved_cfg, dim_pose)

    model_factory, resolved_model_variant = resolve_mmdit_factory(
        _resolve_value(cfg, saved_cfg, "model_variant", "MMDiT"),
    )
    print(
        f"Loading {resolved_model_variant} with "
        f"input_dim={dim_pose} ({motion_representation})"
    )
    ema_model = model_factory(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema_model" not in checkpoint:
        raise KeyError("Could not find EMA model state key in checkpoint.")
    missing_keys, unexpected_keys = ema_model.load_state_dict(
        checkpoint["ema_model"], strict=False
    )
    _validate_checkpoint_keys(missing_keys, unexpected_keys)

    requested_gpu = str(cfg.gpu)
    if requested_gpu != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{requested_gpu}")
    else:
        if requested_gpu != "cpu":
            print("CUDA is unavailable, falling back to CPU.")
        device = torch.device("cpu")

    ema_model.eval()
    ema_model.to(device)

    out_dir = pjoin(cfg.out_dir_root, cfg.exp)
    os.makedirs(out_dir, exist_ok=True)

    clip_text, frame_lengths = load_prompts_and_lengths(cfg)

    if cfg.get("prompt_csv"):
        expanded_texts: list[str] = []
        expanded_lengths: list[int] = []
        for text, length in zip(clip_text, frame_lengths):
            expanded_texts.extend([text] * cfg.num_samples)
            expanded_lengths.extend([length] * cfg.num_samples)
        clip_text = expanded_texts
        frame_lengths = expanded_lengths

    max_length = int(max(frame_lengths))
    if max_length > int(model_kwargs["max_length"]):
        raise ValueError(
            f"Requested max length {max_length} exceeds model max_length "
            f"{model_kwargs['max_length']}."
        )

    m_length = torch.tensor(frame_lengths, dtype=torch.long, device=device)

    with torch.no_grad():
        pred_motions = ema_model.generate(clip_text, m_length, cond_scale=cfg.cfg)

    if pred_motions.ndim == 4:
        pred_motions = pred_motions[:, :, 0, :]
    elif pred_motions.ndim != 3:
        raise ValueError(
            f"Unexpected generated motion shape: {tuple(pred_motions.shape)}"
        )

    normalized_motions = pred_motions.detach().cpu().numpy()
    if raw_joint_shape is None:
        if normalized_motions.shape[-1] < 67:
            raise ValueError(
                "Generated motion must contain at least the first 67 HumanML3D "
                f"features to recover joints, got {normalized_motions.shape[-1]}."
            )
        pred_motions_np = (
            normalized_motions * std[: normalized_motions.shape[-1]]
            + mean[: normalized_motions.shape[-1]]
        )
        joints = recover_from_ric(
            torch.from_numpy(pred_motions_np), joints_num=22
        ).numpy()
    else:
        raw_joint_count, raw_joint_dim = raw_joint_shape
        normalized_joints = unflatten_raw_joints_np(
            normalized_motions,
            joint_count=raw_joint_count,
            joint_dim=raw_joint_dim,
        )
        joints = normalized_joints * std + mean
        pred_motions_np = joints

    print(f"Generated {motion_representation} shape:", pred_motions_np.shape)
    print("Recovered joints shape:", joints.shape)

    for i, (text, length) in enumerate(zip(clip_text, frame_lengths)):
        pair_idx = i // cfg.num_samples
        rep_idx = i % cfg.num_samples
        sample_name = f"sample_{pair_idx:02d}_rep{rep_idx:02d}"

        motion_slice = pred_motions_np[i, :length]
        joints_slice = joints[i, :length]

        np.save(os.path.join(out_dir, f"{sample_name}_motion.npy"), motion_slice)
        np.save(os.path.join(out_dir, f"{sample_name}.npy"), joints_slice)

        output_path = os.path.join(out_dir, f"{sample_name}.mp4")
        plot_3d_motion(output_path, joints_slice, str(text), fps=20, radius=4)
        print(f"Saved sample {i} to: {output_path}")

    print(f"All samples saved to: {out_dir}")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="demo_msflow_263",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
