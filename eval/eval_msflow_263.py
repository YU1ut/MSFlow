import os
import random
import re
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils.config_utils import apply_config_to_cfg, load_yaml_config
from utils.datasets import Text2MotionDataset, collate_fn
from utils.eval_msflow_263 import evaluation_msflow_263
from utils.eval_msflow_xyz import evaluation_msflow_xyz
from utils.evaluators import Evaluators
from utils.mmdit_model_utils import resolve_mmdit_factory
from utils.raw_joint_utils import load_raw_joint_mean_std


def _resolve_checkpoint_file(model_dir: str, checkpoint_name: str) -> str:
    checkpoint_file = (
        checkpoint_name
        if checkpoint_name.endswith(".tar")
        else f"{checkpoint_name}.tar"
    )
    return pjoin(model_dir, checkpoint_file)


def _resolve_ema_key(checkpoint: dict) -> str:
    if "ema_model" not in checkpoint:
        raise KeyError("Could not find EMA model state key in checkpoint.")
    return "ema_model"


def _format_cond_scale_for_path(cond_scale: object) -> str:
    token = str(cond_scale).strip().replace("-", "m").replace(".", "p")
    token = re.sub(r"[^A-Za-z0-9_]+", "_", token).strip("_")
    return token or "none"


def _get_explicit_task_override(cfg: DictConfig, key: str) -> tuple[bool, object]:
    try:
        task_overrides = HydraConfig.get().overrides.task
    except ValueError:
        return False, None

    prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    for override in task_overrides:
        if override.startswith(prefixes):
            return True, cfg.get(key)
    return False, None


def run(cfg: DictConfig) -> None:
    has_sampler_override, sampler_override = _get_explicit_task_override(cfg, "sampler")
    has_feature_dim_override, _ = _get_explicit_task_override(cfg, "feature_dim")
    dit_config_path = pjoin(
        cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "config.yaml"
    )
    dit_config = load_yaml_config(dit_config_path)
    if dit_config:
        cfg = apply_config_to_cfg(
            cfg,
            dit_config,
            (
                "dataset_name",
                "dataset_dir",
                "max_motion_length",
                "unit_length",
                "raw_joint_count",
                "raw_joint_dim",
                "raw_mean_path",
                "raw_std_path",
                "raw_normalization",
                "raw_joint_layout",
                "patch_size",
                "stride_size",
                "motion_fps",
                "max_length",
                "is_causal",
                "num_frame_per_block",
                "use_single_token_refiner",
                "bert_model_path",
                "noise_scale",
                "flow_output_type",
                "flow_t_sampler",
                "model_variant",
            ),
        )
        if "eval_feature_dim" in dit_config and not has_feature_dim_override:
            cfg.feature_dim = dit_config["eval_feature_dim"]
        print(f"Loaded DiT config from {dit_config_path}")
    else:
        print(f"No saved DiT config found at {dit_config_path}. Using Hydra config.")

    if has_sampler_override:
        cfg.sampler = sampler_override
        print(f"Using sampler override: {cfg.sampler}")

    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    if cfg.dataset_name != "t2m":
        raise NotImplementedError(f"Unsupported dataset: {cfg.dataset_name}")

    humanml_data_root = f"{cfg.dataset_dir}/HumanML3D/"
    data_root = humanml_data_root
    split_dir = data_root
    motion_fps = int(cfg.get("motion_fps", 20))
    raw_joint_count_value = cfg.get("raw_joint_count")
    motion_feature_dim = None
    if raw_joint_count_value is None:
        raw_joint_count = None
        raw_joint_dim = 3
        dim_pose = 263
        motion_feature_dim = dim_pose
        motion_dir = pjoin(data_root, "new_joint_vecs")
        mean = np.load(pjoin(data_root, "Mean.npy"))
        std = np.load(pjoin(data_root, "Std.npy"))
        motion_representation = "HumanML3D features"
    else:
        raw_joint_count = int(raw_joint_count_value)
        raw_joint_dim = int(cfg.get("raw_joint_dim", 3))
        if raw_joint_count != 22 or raw_joint_dim != 3:
            raise ValueError(
                "MMDiT_xyz evaluation expects 22 XYZ joints, got "
                f"raw_joint_count={raw_joint_count} and "
                f"raw_joint_dim={raw_joint_dim}."
            )
        if int(cfg.feature_dim) != 67:
            raise ValueError("Raw XYZ evaluation requires feature_dim=67.")
        dim_pose = raw_joint_count * raw_joint_dim
        motion_dir = pjoin(data_root, "new_joints")
        raw_normalization = cfg.get("raw_normalization", "shared_xyz")
        mean, std = load_raw_joint_mean_std(
            cfg.dataset_name,
            data_root=data_root,
            mean_path=cfg.get("raw_mean_path"),
            std_path=cfg.get("raw_std_path"),
            normalization=raw_normalization,
            joint_count=raw_joint_count,
            joint_dim=raw_joint_dim,
            split_file=pjoin(data_root, "train.txt"),
        )
        motion_representation = f"raw XYZ joints ({raw_normalization} normalization)"

    text_dir = pjoin(data_root, "texts")

    if cfg.feature_dim != 67:
        raise ValueError("The retained evaluator requires feature_dim=67.")
    eval_mean = np.load(f"utils/eval_mean_std/{cfg.dataset_name}/eval_mean_67.npy")
    eval_std = np.load(f"utils/eval_mean_std/{cfg.dataset_name}/eval_std_67.npy")

    split_file = pjoin(split_dir, "test.txt")
    dataset_kwargs = {
        "feature_dim": motion_feature_dim,
        "motion_fps": motion_fps,
    }
    eval_dataset = Text2MotionDataset(
        mean,
        std,
        split_file,
        cfg.dataset_name,
        motion_dir,
        text_dir,
        cfg.unit_length,
        cfg.max_motion_length,
        20,
        evaluation=True,
        **dataset_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=True,
    )

    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, "model")

    model_factory, resolved_model_variant = resolve_mmdit_factory(
        cfg.get("model_variant", "MMDiT"),
    )
    model_kwargs = {
        "input_dim": dim_pose,
        "max_length": cfg.max_length,
        "is_causal": cfg.is_causal,
        "num_frame_per_block": cfg.num_frame_per_block,
        "use_single_token_refiner": cfg.use_single_token_refiner,
        "bert_model_path": cfg.bert_model_path,
        "noise_scale": cfg.noise_scale,
        "flow_output_type": cfg.flow_output_type,
        "flow_t_sampler": cfg.flow_t_sampler,
        "sigma_min": cfg.sigma_min,
        "sampler": cfg.sampler,
    }
    if raw_joint_count is not None:
        model_kwargs.update(
            {
                "raw_joint_dim": raw_joint_dim,
                "raw_joint_count": raw_joint_count,
                "patch_size": tuple(cfg.patch_size),
                "stride_size": tuple(cfg.stride_size),
            }
        )

    print(
        f"Loading {resolved_model_variant} with "
        f"input_dim={dim_pose} ({motion_representation})"
    )
    ema_model = model_factory(**model_kwargs)
    checkpoint_path = _resolve_checkpoint_file(model_dir, cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ema_key = _resolve_ema_key(checkpoint)
    missing_keys, unexpected_keys = ema_model.load_state_dict(
        checkpoint[ema_key], strict=False
    )
    unsupported_missing_keys = [
        key for key in missing_keys if not key.startswith(("clip_model.",))
    ]
    if unexpected_keys or unsupported_missing_keys:
        details = []
        if unexpected_keys:
            details.append(f"unexpected keys: {', '.join(unexpected_keys[:20])}")
        if unsupported_missing_keys:
            details.append(f"missing keys: {', '.join(unsupported_missing_keys[:20])}")
        raise RuntimeError("Could not load EMA checkpoint: " + "; ".join(details))
    if missing_keys:
        print(
            f"Loaded EMA checkpoint with {len(missing_keys)} optional "
            "encoder/conditioning keys missing."
        )

    device = torch.device(f"cuda:{cfg.gpu}" if cfg.gpu != "cpu" else "cpu")
    print(f"Evaluator feature dimension: {cfg.feature_dim}")
    eval_wrapper = Evaluators(
        cfg.dataset_name, device=device, feature_dim=cfg.feature_dim
    )

    #################################################################################
    #                                    Evaluation Loop                            #
    #################################################################################
    eval_dir_name = (
        f"eval_raw_{cfg.feature_dim}"
        if raw_joint_count is not None
        else f"eval_{cfg.feature_dim}"
    )
    out_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name, eval_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    cond_scale = cfg.cfg
    cond_scale_token = _format_cond_scale_for_path(cond_scale)
    log_suffix = "_mm" if cfg.cal_mm else ""
    log_path = pjoin(out_dir, f"eval_cond_scale_{cond_scale_token}{log_suffix}.log")

    ema_model.eval()
    ema_model.to(device)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mm = []
    clip_scores = []

    evaluation_fn = (
        evaluation_msflow_xyz if raw_joint_count is not None else evaluation_msflow_263
    )
    evaluation_kwargs: dict[str, object] = {}
    if raw_joint_count is not None:
        evaluation_kwargs = {
            "dataset_name": cfg.dataset_name,
            "raw_joint_count": raw_joint_count,
            "raw_joint_dim": raw_joint_dim,
        }

    repeat_time = cfg.repeat_time
    for i in range(repeat_time):
        with torch.no_grad():
            (
                best_fid,
                best_div,
                best_top1,
                best_top2,
                best_top3,
                best_matching,
                best_mm,
                clip_score,
            ) = (1000, 0, 0, 0, 0, 100, 0, -1)
            (
                best_fid,
                best_div,
                best_top1,
                best_top2,
                best_top3,
                best_matching,
                best_mm,
                clip_score,
            ), _ = evaluation_fn(
                eval_loader,
                ema_model,
                i,
                best_fid=best_fid,
                clip_score_old=clip_score,
                best_div=best_div,
                best_top1=best_top1,
                best_top2=best_top2,
                best_top3=best_top3,
                best_matching=best_matching,
                eval_wrapper=eval_wrapper,
                device=device,
                eval_mean=eval_mean,
                eval_std=eval_std,
                cond_scale=cond_scale,
                cal_mm=cfg.cal_mm,
                return_current=True,
                **evaluation_kwargs,
            )

        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        mm.append(best_mm)
        clip_scores.append(clip_score)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mm = np.array(mm)
    clip_scores = np.array(clip_scores)

    print("final result:")
    confidence_scale = 1.96 / np.sqrt(repeat_time)
    msg_final = (
        f"\tFID: {np.mean(fid):.3f}, "
        f"conf. {np.std(fid) * confidence_scale:.3f}\n"
        f"\tDiversity: {np.mean(div):.3f}, "
        f"conf. {np.std(div) * confidence_scale:.3f}\n"
        f"\tTOP1: {np.mean(top1):.3f}, "
        f"conf. {np.std(top1) * confidence_scale:.3f}, "
        f"TOP2. {np.mean(top2):.3f}, "
        f"conf. {np.std(top2) * confidence_scale:.3f}, "
        f"TOP3. {np.mean(top3):.3f}, "
        f"conf. {np.std(top3) * confidence_scale:.3f}\n"
        f"\tMatching: {np.mean(matching):.3f}, "
        f"conf. {np.std(matching) * confidence_scale:.3f}\n"
        f"\tMultimodality:{np.mean(mm):.3f}, "
        f"conf.{np.std(mm) * confidence_scale:.3f}\n\n"
        f"\tCLIP-Score:{np.mean(clip_scores):.3f}, "
        f"conf.{np.std(clip_scores) * confidence_scale:.3f}\n"
    )
    print(msg_final)
    with open(log_path, "a", encoding="utf-8") as log_file:
        print(f"cond_scale: {cond_scale}", file=log_file, flush=True)
        print("final result:", file=log_file, flush=True)
        print(msg_final, file=log_file, flush=True)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="eval_msflow_263",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
