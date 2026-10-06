from __future__ import annotations

import argparse
import json
import os
from os.path import join as pjoin
from typing import Any, Sequence

import numpy as np
import torch

from utils.plot_script import plot_3d_motion

SPARSE_CONTROL_STAGE_COLORS = {
    "early": "#0072B2",
    "middle": "#CC79A7",
    "late": "#009E73",
}


def normalize_world(
    motion: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (motion - mean) / std


def unnormalize_motion(
    motion: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return motion * std + mean


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


def parse_axes(spec: str) -> tuple[int, ...]:
    spec = (spec or "x,y,z").replace(" ", "")
    mapping = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}
    try:
        axes = [mapping[item] for item in spec.split(",") if item]
    except KeyError as exc:
        raise ValueError(f"Unknown axis in --axes '{spec}'. Use x,y,z.") from exc
    return tuple(dict.fromkeys(axes))


def parse_joint_ids(spec: str | int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(spec, int):
        return (int(spec),)
    if isinstance(spec, (list, tuple, np.ndarray)):
        ids = [int(item) for item in spec]
    else:
        ids = [int(item) for item in str(spec).replace(" ", "").split(",") if item]
    if not ids:
        raise ValueError("At least one joint id is required.")
    return tuple(dict.fromkeys(ids))


def resolve_value(
    args: argparse.Namespace,
    saved_cfg: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    arg_val = getattr(args, key, None)
    if arg_val is not None:
        return arg_val
    saved_val = saved_cfg.get(key)
    if saved_val is not None:
        return saved_val
    return default


def load_sparse_raw_control_spec(
    path: str,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    n_joints: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], str, dict[str, Any]]:
    """Load embedded arbitrary frame/joint targets from a JSON specification."""
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    motion_id = str(spec.get("source_motion_id", ""))
    prompt = str(spec["prompt"])
    n_frames = int(spec["n_frames"])
    if n_frames <= 0:
        raise ValueError("Sparse control spec n_frames must be positive.")

    controls = spec.get("controls", [])
    if not controls:
        raise ValueError("Sparse control spec must contain at least one control.")

    mask = np.zeros((1, 3, n_frames, n_joints), dtype=np.float32)
    control_world = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
    controlled_joint_ids: list[int] = []
    resolved_controls = []
    for entry in controls:
        has_frame = "frame" in entry
        has_fraction = "frame_fraction" in entry
        if has_frame == has_fraction:
            raise ValueError(
                "Each sparse control needs exactly one of frame or frame_fraction."
            )
        if has_frame:
            frame = int(entry["frame"])
        else:
            fraction = float(entry["frame_fraction"])
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError("frame_fraction must be in [0, 1].")
            frame = int(round(fraction * (n_frames - 1)))
        if frame < 0 or frame >= n_frames:
            raise ValueError(f"Control frame must be in [0, {n_frames}), got {frame}.")

        joint_ids = parse_joint_ids(entry["joints"])
        if any(joint_id < 0 or joint_id >= n_joints for joint_id in joint_ids):
            raise ValueError(f"Joint ids must be in [0, {n_joints}), got {joint_ids}.")
        axes = parse_axes(str(entry.get("axes", "x,y,z")))
        values = np.asarray(entry["values"], dtype=np.float32)
        expected_shape = (len(joint_ids), 3)
        if values.shape != expected_shape:
            raise ValueError(
                f"Control values must have shape {expected_shape}, got "
                f"{values.shape}."
            )
        for axis in axes:
            mask[0, axis, frame, list(joint_ids)] = 1.0
        control_world[frame, list(joint_ids)] = values
        controlled_joint_ids.extend(joint_ids)
        resolved_controls.append(
            {
                "stage": str(entry.get("stage", "")),
                "frame": frame,
                "time_seconds": frame / 20.0,
                "joints": list(joint_ids),
                "axes": ["xyz"[axis] for axis in axes],
            }
        )

    control_norm = normalize_world(control_world, mean, std)
    control = control_norm.transpose(2, 0, 1)[None].astype(np.float32)
    metadata = {
        "source_motion_id": motion_id,
        "control_spec_path": path,
        "prompt": prompt,
        "n_frames": n_frames,
        "fps": 20,
        "controls": resolved_controls,
        "stage_colors": SPARSE_CONTROL_STAGE_COLORS,
    }
    return (
        control,
        mask,
        control_world,
        tuple(dict.fromkeys(controlled_joint_ids)),
        prompt,
        metadata,
    )


def sparse_control_hint_colors(
    control_world: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    colors = np.full(control_world.shape[:2], "#80B79A", dtype=object)
    for control in metadata["controls"]:
        colors[control["frame"], control["joints"]] = SPARSE_CONTROL_STAGE_COLORS[
            control["stage"]
        ]
    return colors


def raw_control_chw_to_dit(
    control: np.ndarray,
    mask: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    control_bt1d = control.transpose(0, 2, 3, 1).reshape(
        control.shape[0],
        control.shape[2],
        1,
        control.shape[3] * control.shape[1],
    )
    mask_bt1d = mask.transpose(0, 2, 3, 1).reshape(
        mask.shape[0],
        mask.shape[2],
        1,
        mask.shape[3] * mask.shape[1],
    )
    return torch.from_numpy(control_bt1d), torch.from_numpy(mask_bt1d)


def dit_norm_to_world(
    samples_norm: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    n_joints: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples_flat = samples_norm[:, :, 0, :].detach().cpu().numpy()
    samples_tj3_norm = samples_flat.reshape(samples_flat.shape[0], -1, n_joints, 3)
    samples_world = np.stack(
        [
            unnormalize_motion(sample, mean, std)
            for sample in samples_tj3_norm.astype(np.float32)
        ],
        axis=0,
    )
    return samples_tj3_norm.astype(np.float32), samples_world.astype(np.float32)


def save_raw_inpaint_outputs(
    args: argparse.Namespace,
    *,
    samples_norm_tj3: np.ndarray,
    samples_world: np.ndarray,
    control_world: np.ndarray,
    control_mask_chw: np.ndarray,
    trajectory_requested: np.ndarray,
    trajectory_used: np.ndarray,
    joint_ids: Sequence[int],
    test_motion_id: str | None,
    test_prompt_index: int | None,
    sparse_control_metadata: dict[str, Any] | None = None,
) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(
        pjoin(args.out_dir, "requested_trajectory_xyz.npy"),
        np.asarray(trajectory_requested, dtype=np.float32),
    )
    np.save(
        pjoin(args.out_dir, "trajectory_xyz.npy"),
        np.asarray(trajectory_used, dtype=np.float32),
    )
    np.save(
        pjoin(args.out_dir, "raw_joint_control_world.npy"),
        control_world.astype(np.float32),
    )
    np.save(
        pjoin(args.out_dir, "raw_joint_control_mask_chw.npy"),
        control_mask_chw.astype(np.float32),
    )
    np.save(pjoin(args.out_dir, "motion_raw_norm.npy"), samples_norm_tj3)
    np.save(pjoin(args.out_dir, "samples_world.npy"), samples_world)

    if samples_world.shape[0] == 1:
        np.save(pjoin(args.out_dir, "sample_00_raw_norm.npy"), samples_norm_tj3[0])
        np.save(pjoin(args.out_dir, "sample_00.npy"), samples_world[0])

    if test_motion_id is not None:
        with open(
            pjoin(args.out_dir, "test_trajectory_raw_source.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "motion_id": test_motion_id,
                    "split": args.test_split,
                    "start_frame": int(args.test_motion_start_frame),
                    "n_frames": int(args.n_frames),
                    "prompt_index": int(test_prompt_index),
                    "prompt": args.text,
                },
                f,
                indent=2,
            )

    if sparse_control_metadata is not None:
        with open(
            pjoin(args.out_dir, "sparse_control_source.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(sparse_control_metadata, f, indent=2)

    hint_colors = (
        sparse_control_hint_colors(control_world, sparse_control_metadata)
        if sparse_control_metadata is not None
        else None
    )

    _save_xz_plot(
        pjoin(args.out_dir, "xz_path.png"),
        trajectory_used,
        samples_world,
        joint_ids,
    )

    if args.save_mp4 or args.save_gif:
        hint = control_world
        for idx in range(samples_world.shape[0]):
            sample_name = f"sample_{idx:02d}"
            if args.save_mp4:
                plot_3d_motion(
                    pjoin(args.out_dir, f"{sample_name}.mp4"),
                    samples_world[idx],
                    args.text,
                    fps=20,
                    radius=4,
                    hint=hint,
                    hint_colors=hint_colors,
                )
            if args.save_gif:
                plot_3d_motion(
                    pjoin(args.out_dir, f"{sample_name}.gif"),
                    samples_world[idx],
                    args.text,
                    fps=20,
                    radius=4,
                    hint=hint,
                    hint_colors=hint_colors,
                )

    print("Saved:")
    for name in (
        "requested_trajectory_xyz.npy",
        "trajectory_xyz.npy",
        "raw_joint_control_world.npy",
        "raw_joint_control_mask_chw.npy",
        "motion_raw_norm.npy",
        "samples_world.npy",
        "xz_path.png",
    ):
        print(" -", pjoin(args.out_dir, name))
    if samples_world.shape[0] == 1:
        print(" -", pjoin(args.out_dir, "sample_00_raw_norm.npy"))
        print(" -", pjoin(args.out_dir, "sample_00.npy"))
    if test_motion_id is not None:
        print(" -", pjoin(args.out_dir, "test_trajectory_raw_source.json"))
    if sparse_control_metadata is not None:
        print(" -", pjoin(args.out_dir, "sparse_control_source.json"))
    if args.save_mp4:
        print(" -", pjoin(args.out_dir, "sample_00.mp4"))
    if args.save_gif:
        print(" -", pjoin(args.out_dir, "sample_00.gif"))


def _trajectory_for_plot(trajectory: np.ndarray) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim == 2:
        return traj[:, None, :]
    if traj.ndim == 3:
        return traj
    raise ValueError(f"Unsupported trajectory shape for plotting: {traj.shape}")


def _save_xz_plot(
    path: str,
    trajectory: np.ndarray,
    samples_world: np.ndarray,
    joint_ids: Sequence[int],
) -> None:
    try:
        import matplotlib.pyplot as plt

        trajectory_plot = _trajectory_for_plot(trajectory)
        plt.figure(figsize=(6, 6))
        for local_idx, joint_id in enumerate(joint_ids):
            if local_idx < trajectory_plot.shape[1]:
                target = trajectory_plot[:, local_idx, :]
            else:
                target = trajectory_plot[:, 0, :]
            plt.plot(
                target[:, 0],
                target[:, 2],
                label=f"target j{int(joint_id)}",
            )
        for sample_idx in range(samples_world.shape[0]):
            for joint_id in joint_ids:
                joint = samples_world[sample_idx, :, int(joint_id), :]
                plt.plot(
                    joint[:, 0],
                    joint[:, 2],
                    label=f"sample {sample_idx:02d} j{int(joint_id)}",
                )
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception:
        pass
