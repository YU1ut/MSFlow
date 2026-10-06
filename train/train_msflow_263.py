import copy
import os
import random
import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

import hydra
import numpy as np
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils.config_utils import save_args_yaml
from utils.datasets import Text2MotionDataset, collate_fn
from utils.eval_msflow_263 import evaluation_msflow_263
from utils.eval_msflow_xyz import evaluation_msflow_xyz
from utils.evaluators import Evaluators
from utils.mmdit_model_utils import resolve_mmdit_factory
from utils.raw_joint_utils import (
    flatten_raw_joints,
    load_raw_joint_mean_std,
    normalize_raw_joint_normalization,
)
from utils.train_utils import (
    def_value,
    print_current_loss,
    save,
    update_ema,
    update_lr_warm_up,
)
from utils.wandb_utils import namespace_to_dict


def _load_eval_mean_std(
    dataset_name: str, feature_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    if feature_dim != 67:
        raise ValueError("eval_feature_dim must be 67.")

    eval_mean = np.load(f"utils/eval_mean_std/{dataset_name}/eval_mean_67.npy")
    eval_std = np.load(f"utils/eval_mean_std/{dataset_name}/eval_std_67.npy")
    return eval_mean, eval_std


def _prepare_motion(
    motion: torch.Tensor,
    raw_joint_count: int | None,
    raw_joint_dim: int,
    raw_joint_layout: bool = False,
) -> torch.Tensor:
    if raw_joint_count is None:
        return motion
    if raw_joint_layout:
        if tuple(motion.shape[-2:]) != (raw_joint_count, raw_joint_dim):
            raise ValueError(
                f"Expected raw motion [B, T, {raw_joint_count}, {raw_joint_dim}], "
                f"got {tuple(motion.shape)}."
            )
        return motion.permute(0, 3, 1, 2).contiguous()
    return flatten_raw_joints(
        motion,
        joint_count=raw_joint_count,
        joint_dim=raw_joint_dim,
    )


def _move_condition_to_device(condition, device):
    if torch.is_tensor(condition):
        return condition.to(device)
    return condition


def _slice_condition(condition, batch_slice):
    return condition[batch_slice]


def run(cfg: DictConfig) -> None:
    eval_every = int(cfg.eval_every)
    if eval_every <= 0:
        raise ValueError("eval_every must be a positive integer.")

    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    wandb_mode = "disabled" if cfg.entity == "your_wandb_entity" else "online"
    wandb.init(
        name=cfg.name,
        project=cfg.project_name,
        entity=cfg.entity,
        config=namespace_to_dict(cfg),
        mode=wandb_mode,
    )

    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    if cfg.dataset_name != "t2m":
        raise NotImplementedError(f"Unsupported dataset: {cfg.dataset_name}")

    humanml_data_root = f"{cfg.dataset_dir}/HumanML3D/"
    data_root = humanml_data_root
    split_dir = data_root
    motion_fps = int(cfg.get("motion_fps", 20))
    raw_joint_count_value = cfg.get("raw_joint_count")
    raw_joint_layout = bool(cfg.get("raw_joint_layout", False))
    motion_feature_dim = None
    raw_normalization = None
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
                "MMDiT_xyz training expects 22 XYZ joints, got "
                f"raw_joint_count={raw_joint_count} and "
                f"raw_joint_dim={raw_joint_dim}."
            )
        dim_pose = raw_joint_count * raw_joint_dim
        motion_dir = pjoin(data_root, "new_joints")
        raw_normalization = normalize_raw_joint_normalization(
            cfg.get("raw_normalization", "shared_xyz")
        )
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
    train_split_file = pjoin(split_dir, "train.txt")
    val_split_file = pjoin(split_dir, "val.txt")

    train_dataset = Text2MotionDataset(
        mean,
        std,
        train_split_file,
        cfg.dataset_name,
        motion_dir,
        text_dir,
        cfg.unit_length,
        cfg.max_motion_length,
        20,
        evaluation=False,
        feature_dim=motion_feature_dim,
        motion_fps=motion_fps,
    )
    val_dataset = Text2MotionDataset(
        mean,
        std,
        val_split_file,
        cfg.dataset_name,
        motion_dir,
        text_dir,
        cfg.unit_length,
        cfg.max_motion_length,
        20,
        evaluation=False,
        feature_dim=motion_feature_dim,
        motion_fps=motion_fps,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=cfg.num_workers,
        shuffle=True,
    )

    eval_feature_dim = int(cfg.get("eval_feature_dim", 67))
    eval_loader = None
    eval_mean = None
    eval_std = None
    if cfg.enable_eval_logging:
        if eval_feature_dim != 67:
            raise ValueError("The retained evaluator requires eval_feature_dim=67.")
        eval_mean, eval_std = _load_eval_mean_std(cfg.dataset_name, eval_feature_dim)
        eval_split_file = pjoin(split_dir, f"{cfg.eval_split}.txt")
        eval_dataset_kwargs = {
            "motion_fps": motion_fps,
            "feature_dim": motion_feature_dim,
        }
        eval_dataset = Text2MotionDataset(
            mean,
            std,
            eval_split_file,
            cfg.dataset_name,
            motion_dir,
            text_dir,
            cfg.unit_length,
            cfg.max_motion_length,
            20,
            evaluation=True,
            **eval_dataset_kwargs,
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
    run_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name)
    model_dir = pjoin(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    if raw_normalization == "per_coordinate":
        raw_mean_path = pjoin(run_dir, "raw_per_coordinate_mean.npy")
        raw_std_path = pjoin(run_dir, "raw_per_coordinate_std.npy")
        np.save(raw_mean_path, mean)
        np.save(raw_std_path, std)
        cfg.raw_mean_path = raw_mean_path
        cfg.raw_std_path = raw_std_path
        cfg.raw_normalization = raw_normalization
    save_args_yaml(cfg, pjoin(run_dir, "config.yaml"))

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
        f"Creating {resolved_model_variant} with "
        f"input_dim={dim_pose} ({motion_representation})"
    )
    model = model_factory(**model_kwargs)
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    all_params = sum(
        param.numel()
        for name, param in model.named_parameters()
        if not name.startswith("clip_model.")
    )
    print(f"Total parameters of MMDiT model: {all_params / 1_000_000:.2f}M")

    device = torch.device(f"cuda:{cfg.gpu}" if cfg.gpu != "cpu" else "cpu")

    eval_wrapper = None
    if cfg.enable_eval_logging:
        eval_wrapper = Evaluators(
            cfg.dataset_name, device=device, feature_dim=eval_feature_dim
        )

    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    model.to(device)
    ema_model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.99),
        lr=cfg.lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.milestones,
        gamma=cfg.lr_decay,
    )

    epoch = 0
    it = 0
    if cfg.batch_size % cfg.accum != 0:
        raise ValueError("batch_size must be divisible by accum.")

    start_time = time.time()
    total_iters = cfg.epoch * len(train_loader)
    print(f"Total Epochs: {cfg.epoch}, Total Iters: {total_iters}")
    print(
        "Iters Per Epoch, Training: %04d, Validation: %03d"
        % (len(train_loader), len(val_loader))
    )

    logs = defaultdict(def_value, OrderedDict())
    best_val_loss = float("inf")
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching = (
        1000,
        0,
        0,
        0,
        0,
        100,
    )
    best_mm = 0
    best_clip_score = -1

    while epoch < cfg.epoch:
        model.train()
        optimizer.zero_grad()

        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < cfg.warm_up_iter:
                update_lr_warm_up(it, cfg.warm_up_iter, optimizer, cfg.lr)

            conds, motion, m_lens = batch_data
            motion = motion.detach().float().to(device)
            motion = _prepare_motion(
                motion,
                raw_joint_count,
                raw_joint_dim,
                raw_joint_layout,
            )
            m_lens = m_lens.detach().long().to(device)
            conds = _move_condition_to_device(conds, device)

            accum = cfg.accum
            accum_bs = cfg.batch_size // accum
            for kk in range(accum):
                batch_slice = slice(kk * accum_bs, (kk + 1) * accum_bs)
                loss = model.forward_loss(
                    motion[batch_slice],
                    _slice_condition(conds, batch_slice),
                    m_lens[batch_slice],
                )
                loss = loss / accum
                loss.backward()
                logs["loss"] += loss.item()
                logs["lr"] += optimizer.param_groups[0]["lr"] / accum

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            update_ema(model, ema_model, 0.9999)

            if it % cfg.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    wandb.log({f"Train/{tag}": value / cfg.log_every}, step=it)
                    mean_loss[tag] = value / cfg.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(
                    start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i
                )

        latest_ckpt_path = pjoin(model_dir, "latest.tar")
        save(
            latest_ckpt_path,
            epoch,
            model,
            optimizer,
            scheduler,
            it,
            "model",
            ema_model,
        )
        if (
            cfg.get("save_eval_every_checkpoint", False)
            and (epoch + 1) % eval_every == 0
        ):
            eval_ckpt_path = pjoin(model_dir, f"epoch_{epoch + 1:04d}.tar")
            save(
                eval_ckpt_path,
                epoch,
                model,
                optimizer,
                scheduler,
                it,
                "model",
                ema_model,
            )
            print(f"Saved eval-interval checkpoint to {eval_ckpt_path}")

        ########################################################################
        #                              Eval Loop                               #
        ########################################################################
        print("Validation time:")
        model.eval()
        val_loss = []
        with torch.no_grad():
            for _, batch_data in enumerate(val_loader):
                conds, motion, m_lens = batch_data
                motion = motion.detach().float().to(device)
                motion = _prepare_motion(
                    motion,
                    raw_joint_count,
                    raw_joint_dim,
                    raw_joint_layout,
                )
                m_lens = m_lens.detach().long().to(device)
                conds = _move_condition_to_device(conds, device)

                loss = model.forward_loss(motion, conds, m_lens)
                val_loss.append(loss.item())

        mean_val_loss = np.mean(val_loss)
        print(f"Validation loss:{mean_val_loss:.3f}")
        wandb.log({"Val/loss": mean_val_loss}, step=it)
        if mean_val_loss < best_val_loss:
            print(f"Improved loss from {best_val_loss:.2f} to {mean_val_loss:.3f}!!!")
            best_val_loss = mean_val_loss
            val_loss_ckpt_path = pjoin(model_dir, "best_val_loss.tar")
            save(
                val_loss_ckpt_path,
                epoch,
                model,
                optimizer,
                scheduler,
                it,
                "model",
                ema=ema_model,
            )
            print(
                f"Saved best validation-loss checkpoint to {val_loss_ckpt_path} "
                f"(Val/loss: {best_val_loss:.4f})"
            )

        if cfg.enable_eval_logging and (epoch + 1) % eval_every == 0:
            print("Generation evaluation time:")
            prev_best_fid = best_fid
            evaluation_fn = (
                evaluation_msflow_xyz
                if raw_joint_count is not None
                else evaluation_msflow_263
            )
            raw_eval_kwargs = {}
            if raw_joint_count is not None:
                raw_eval_kwargs = {
                    "dataset_name": cfg.dataset_name,
                    "raw_joint_count": raw_joint_count,
                    "raw_joint_dim": raw_joint_dim,
                }
            (
                (
                    best_fid,
                    best_div,
                    best_top1,
                    best_top2,
                    best_top3,
                    best_matching,
                    best_mm,
                    best_clip_score,
                ),
                current_eval_metrics,
            ) = evaluation_fn(
                eval_loader,
                ema_model,
                epoch,
                best_fid=best_fid,
                clip_score_old=best_clip_score,
                best_div=best_div,
                best_top1=best_top1,
                best_top2=best_top2,
                best_top3=best_top3,
                best_matching=best_matching,
                eval_wrapper=eval_wrapper,
                device=device,
                eval_mean=eval_mean,
                eval_std=eval_std,
                cond_scale=cfg.eval_cfg,
                cal_mm=cfg.eval_cal_mm,
                return_current=True,
                **raw_eval_kwargs,
            )
            wandb.log(
                {
                    "Eval/FID": current_eval_metrics["fid"],
                    "Eval/Diversity": current_eval_metrics["diversity"],
                    "Eval/top1": current_eval_metrics["top1"],
                    "Eval/top2": current_eval_metrics["top2"],
                    "Eval/top3": current_eval_metrics["top3"],
                    "Eval/matching": current_eval_metrics["matching"],
                    "Eval/multimodality": current_eval_metrics["multimodality"],
                    "Eval/clip_score": current_eval_metrics["clip_score"],
                    "EvalBest/FID": best_fid,
                    "EvalBest/Diversity": best_div,
                    "EvalBest/top1": best_top1,
                    "EvalBest/top2": best_top2,
                    "EvalBest/top3": best_top3,
                    "EvalBest/matching": best_matching,
                    "EvalBest/multimodality": best_mm,
                    "EvalBest/clip_score": best_clip_score,
                },
                step=it,
            )
            if best_fid < prev_best_fid:
                fid_ckpt_path = pjoin(model_dir, "best_fid.tar")
                save(
                    fid_ckpt_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    it,
                    "model",
                    ema=ema_model,
                )
                print(
                    f"Saved best FID checkpoint to {fid_ckpt_path} "
                    f"(FID: {best_fid:.4f})"
                )
        epoch += 1


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train_msflow_263",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
