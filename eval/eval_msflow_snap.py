"""Evaluate the checkpoint-compatible MSFlow model on SnapMoGen."""

import os
import random
import re
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from train.train_msflow_snap import (
    _build_dataset,
    _build_evaluator,
    _build_model,
)
from utils.config_utils import apply_config_to_cfg, load_yaml_config
from utils.datasets_snap import snap_collate_fn
from utils.eval_msflow_snap import evaluation_msflow_snap

_TRAINING_CONFIG_KEYS = (
    "dataset_name",
    "dataset_dir",
    "model_input_dim",
    "max_motion_length",
    "snap_min_motion_length",
    "unit_length",
    "max_length",
    "is_causal",
    "num_frame_per_block",
    "use_single_token_refiner",
    "noise_scale",
    "bert_model_path",
    "snap_text_encoder_path",
    "snap_text_encoder_local_files_only",
    "snap_evaluator_config",
    "snap_evaluator_checkpoint",
    "flow_output_type",
    "flow_t_sampler",
    "sigma_min",
)


def _checkpoint_path(model_dir, checkpoint):
    filename = checkpoint if str(checkpoint).endswith(".tar") else f"{checkpoint}.tar"
    return pjoin(model_dir, filename)


def _scale_token(scale):
    token = str(scale).replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", token).strip("_")


def _format_metric(name, values, repeat_time):
    confidence = np.std(values) * 1.96 / np.sqrt(repeat_time)
    return f"\t{name}: {np.mean(values):.3f}, conf. {confidence:.3f}"


def run(cfg):
    config_path = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "config.yaml")
    training_config = load_yaml_config(config_path)
    if training_config:
        cfg = apply_config_to_cfg(cfg, training_config, _TRAINING_CONFIG_KEYS)
        print(f"Loaded training config from {config_path}")

    if cfg.dataset_name != "snap":
        raise ValueError("This entrypoint requires dataset_name=snap")

    torch.backends.cudnn.benchmark = False
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    eval_dataset, _, _ = _build_dataset(cfg, "test")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.snap_eval_batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=snap_collate_fn,
        shuffle=True,
    )

    model = _build_model(cfg)
    model_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "model")
    checkpoint_path = _checkpoint_path(model_dir, cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["ema_model"], strict=False
    )
    unsupported_missing_keys = [
        key for key in missing_keys if not key.startswith("clip_model.")
    ]
    if unexpected_keys or unsupported_missing_keys:
        details = []
        if unsupported_missing_keys:
            details.append("missing keys: " + ", ".join(unsupported_missing_keys[:20]))
        if unexpected_keys:
            details.append("unexpected keys: " + ", ".join(unexpected_keys[:20]))
        raise RuntimeError("Could not load EMA checkpoint: " + "; ".join(details))
    if missing_keys:
        print(
            "Loaded EMA checkpoint with "
            f"{len(missing_keys)} missing frozen text-encoder keys."
        )
    device = torch.device(f"cuda:{cfg.gpu}" if cfg.gpu != "cpu" else "cpu")
    model.to(device).eval()
    evaluator = _build_evaluator(cfg, device)

    metrics = {
        "FID": [],
        "Diversity": [],
        "TOP1": [],
        "TOP2": [],
        "TOP3": [],
        "Matching": [],
        "Multimodality": [],
    }
    for repeat in range(cfg.repeat_time):
        (_, current) = evaluation_msflow_snap(
            checkpoint_path,
            eval_loader,
            model,
            None,
            repeat,
            1000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            evaluator,
            device,
            -1.0,
            cond_scale=cfg.cfg,
            cal_mm=cfg.cal_mm,
            return_current=True,
        )
        metrics["FID"].append(current["fid"])
        metrics["Diversity"].append(current["diversity"])
        metrics["TOP1"].append(current["top1"])
        metrics["TOP2"].append(current["top2"])
        metrics["TOP3"].append(current["top3"])
        metrics["Matching"].append(current["matching"])
        metrics["Multimodality"].append(current["multimodality"])

    lines = [
        _format_metric(name, np.asarray(values), cfg.repeat_time)
        for name, values in metrics.items()
    ]
    result = "final result:\n" + "\n".join(lines)
    print(result)

    out_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "eval_snap")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_mm" if cfg.cal_mm else ""
    log_path = pjoin(
        out_dir,
        f"eval_cond_scale_{_scale_token(cfg.cfg)}{suffix}.log",
    )
    with open(log_path, "a", encoding="utf-8") as log_file:
        print(result, file=log_file, flush=True)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="eval_msflow_snap",
)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
