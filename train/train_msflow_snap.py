"""Train the checkpoint-compatible MSFlow model on SnapMoGen motions."""

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
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import wandb
from models.MMDiT import build_model
from utils.config_utils import save_args_yaml
from utils.datasets_snap import SnapText2MotionDataset, snap_collate_fn
from utils.eval_msflow_snap import evaluation_msflow_snap
from utils.snap_evaluator import SnapMoGenEvaluator
from utils.train_utils import (
    def_value,
    print_current_loss,
    save,
    update_ema,
    update_lr_warm_up,
)
from utils.wandb_utils import namespace_to_dict


def _build_dataset(cfg, split):
    data_root = pjoin(str(cfg.dataset_dir), "SnapMoGen")
    mean = np.load(pjoin(data_root, "meta_data", "mean.npy"))
    std = np.load(pjoin(data_root, "meta_data", "std.npy"))
    dataset = SnapText2MotionDataset(
        mean,
        std,
        pjoin(data_root, "data_split_info", f"{split}_ids.txt"),
        pjoin(data_root, "renamed_feats"),
        pjoin(data_root, "all_caption_clean.json"),
        cfg.unit_length,
        cfg.max_motion_length,
        cfg.snap_min_motion_length,
    )
    return dataset, mean, std


def _build_model(cfg):
    if int(cfg.model_input_dim) != 296:
        raise ValueError("SnapMoGen direct training requires model_input_dim=296")
    return build_model(
        input_dim=cfg.model_input_dim,
        max_length=cfg.max_length,
        is_causal=cfg.is_causal,
        num_frame_per_block=cfg.num_frame_per_block,
        use_single_token_refiner=cfg.use_single_token_refiner,
        bert_model_path=cfg.bert_model_path,
        noise_scale=cfg.noise_scale,
        flow_output_type=cfg.flow_output_type,
        flow_t_sampler=cfg.flow_t_sampler,
        sigma_min=cfg.sigma_min,
        sampler=cfg.sampler,
    )


def _build_evaluator(cfg, device):
    return SnapMoGenEvaluator(
        cfg.snap_evaluator_config,
        cfg.snap_evaluator_checkpoint,
        device,
        text_model_path=cfg.snap_text_encoder_path,
        local_files_only=cfg.snap_text_encoder_local_files_only,
    )


def run(cfg):
    if cfg.dataset_name != "snap":
        raise ValueError("This entrypoint requires dataset_name=snap")
    if cfg.batch_size % cfg.accum != 0:
        raise ValueError("batch_size must be divisible by accum")

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

    train_dataset, _, _ = _build_dataset(cfg, "train")
    val_dataset, _, _ = _build_dataset(cfg, "val")
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

    eval_loader = None
    if cfg.enable_eval_logging:
        eval_dataset, _, _ = _build_dataset(cfg, cfg.eval_split)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=cfg.snap_eval_batch_size,
            num_workers=cfg.num_workers,
            drop_last=True,
            collate_fn=snap_collate_fn,
            shuffle=True,
        )

    run_dir = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name)
    model_dir = pjoin(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    save_args_yaml(cfg, pjoin(run_dir, "config.yaml"))

    model = _build_model(cfg)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    device = torch.device(f"cuda:{cfg.gpu}" if cfg.gpu != "cpu" else "cpu")
    model.to(device)
    ema_model.to(device)
    eval_wrapper = _build_evaluator(cfg, device) if cfg.enable_eval_logging else None

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"Trainable SnapMoGen direct-MM parameters: {trainable_parameters / 1e6:.2f}M"
    )
    optimizer = optim.AdamW(
        model.parameters(), betas=(0.9, 0.99), lr=cfg.lr, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.milestones, gamma=cfg.lr_decay
    )

    epoch = 0
    iteration = 0
    total_iterations = cfg.epoch * len(train_loader)
    start_time = time.time()
    logs = defaultdict(def_value, OrderedDict())
    best_val_loss = float("inf")
    best_fid = 1000.0
    best_diversity = 0.0
    best_top1 = best_top2 = best_top3 = 0.0
    best_matching = -1.0
    best_multimodality = 0.0
    best_clip_score = -1.0
    print(f"Total epochs: {cfg.epoch}, total iterations: {total_iterations}")
    while epoch < cfg.epoch:
        model.train()
        optimizer.zero_grad()
        for inner_iteration, batch in enumerate(train_loader):
            iteration += 1
            if iteration < cfg.warm_up_iter:
                update_lr_warm_up(iteration, cfg.warm_up_iter, optimizer, cfg.lr)

            captions, motions, motion_lengths = batch
            motions = motions.to(device).float()
            motion_lengths = motion_lengths.to(device).long()
            microbatch_size = cfg.batch_size // cfg.accum
            for microbatch in range(cfg.accum):
                start = microbatch * microbatch_size
                end = start + microbatch_size
                loss = (
                    model.forward_loss(
                        motions[start:end],
                        captions[start:end],
                        motion_lengths[start:end],
                    )
                    / cfg.accum
                )
                loss.backward()
                logs["loss"] += loss.item()
                logs["lr"] += optimizer.param_groups[0]["lr"] / cfg.accum

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            update_ema(model, ema_model, 0.9999)

            if iteration % cfg.log_every == 0:
                mean_loss = OrderedDict(
                    (key, value / cfg.log_every) for key, value in logs.items()
                )
                for key, value in mean_loss.items():
                    wandb.log({f"Train/{key}": value}, step=iteration)
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(
                    start_time,
                    iteration,
                    total_iterations,
                    mean_loss,
                    epoch=epoch,
                    inner_iter=inner_iteration,
                )

        latest_path = pjoin(model_dir, "latest.tar")
        save(
            latest_path,
            epoch,
            model,
            optimizer,
            scheduler,
            iteration,
            "model",
            ema_model,
        )

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for captions, motions, motion_lengths in val_loader:
                validation_losses.append(
                    model.forward_loss(
                        motions.to(device).float(),
                        captions,
                        motion_lengths.to(device).long(),
                    ).item()
                )
        validation_loss = float(np.mean(validation_losses))
        wandb.log({"Val/loss": validation_loss}, step=iteration)
        print(f"Validation loss: {validation_loss:.4f}")
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            save(
                pjoin(model_dir, "best_val_loss.tar"),
                epoch,
                model,
                optimizer,
                scheduler,
                iteration,
                "model",
                ema_model,
            )

        if cfg.enable_eval_logging and (epoch + 1) % cfg.eval_every == 0:
            previous_best_fid = best_fid
            (
                (
                    best_fid,
                    best_diversity,
                    best_top1,
                    best_top2,
                    best_top3,
                    best_matching,
                    best_multimodality,
                    best_clip_score,
                ),
                metrics,
            ) = evaluation_msflow_snap(
                model_dir,
                eval_loader,
                ema_model,
                None,
                epoch,
                best_fid,
                best_diversity,
                best_top1,
                best_top2,
                best_top3,
                best_matching,
                eval_wrapper,
                device,
                best_clip_score,
                cond_scale=cfg.eval_cfg,
                cal_mm=cfg.eval_cal_mm,
                return_current=True,
            )
            wandb.log(
                {f"Eval/{key}": value for key, value in metrics.items()},
                step=iteration,
            )
            if best_fid < previous_best_fid:
                save(
                    pjoin(model_dir, "best_fid.tar"),
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    iteration,
                    "model",
                    ema_model,
                )
        epoch += 1

    wandb.finish()


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train_msflow_snap",
)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
