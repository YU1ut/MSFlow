from pathlib import Path

import numpy as np
import torch

_NORMALIZATION_ALIASES = {
    "shared_xyz": "shared_xyz",
    "xyz": "shared_xyz",
    "legacy": "shared_xyz",
    "per_coordinate": "per_coordinate",
    "per_joint_coordinate": "per_coordinate",
    "per_feature": "per_coordinate",
    "66d": "per_coordinate",
}


def normalize_raw_joint_normalization(normalization):
    normalized = str(normalization or "shared_xyz").strip().lower().replace("-", "_")
    if normalized not in _NORMALIZATION_ALIASES:
        expected = ", ".join(("shared_xyz", "per_coordinate"))
        raise ValueError(
            f"Unsupported raw joint normalization {normalization!r}. "
            f"Expected one of: {expected}."
        )
    return _NORMALIZATION_ALIASES[normalized]


def _first_existing_pair(candidates):
    for mean_path, std_path in candidates:
        if mean_path is None or std_path is None:
            continue
        mean_file = Path(mean_path)
        std_file = Path(std_path)
        if mean_file.exists() and std_file.exists():
            return mean_file, std_file
    return None, None


def _validate_raw_joint_mean_std(
    mean,
    std,
    normalization,
    joint_count,
    joint_dim,
):
    mean = np.asarray(mean)
    std = np.asarray(std)
    if normalization == "shared_xyz":
        expected_shape = (joint_dim,)
    else:
        expected_shape = (joint_count, joint_dim)
        flat_shape = (joint_count * joint_dim,)
        if mean.shape == flat_shape:
            mean = mean.reshape(expected_shape)
        if std.shape == flat_shape:
            std = std.reshape(expected_shape)

    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(
            f"Raw joint {normalization} mean/std must both have shape "
            f"{expected_shape}, got {mean.shape} and {std.shape}."
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Raw joint mean/std must contain only finite values.")
    if (std <= 0).any():
        raise ValueError("Raw joint std must contain only positive values.")
    return mean.astype(np.float32), std.astype(np.float32)


def compute_raw_joint_mean_std(
    motion_dir,
    split_file,
    joint_count=22,
    joint_dim=3,
):
    motion_dir = Path(motion_dir)
    split_file = Path(split_file)
    names = [
        name.strip() for name in split_file.read_text().splitlines() if name.strip()
    ]

    total_frames = 0
    motion_sum = np.zeros((joint_count, joint_dim), dtype=np.float64)
    motion_square_sum = np.zeros((joint_count, joint_dim), dtype=np.float64)
    for name in names:
        motion_path = motion_dir / f"{name}.npy"
        if not motion_path.is_file():
            continue
        motion = np.load(motion_path)
        if motion.shape[1:] != (joint_count, joint_dim):
            continue
        if len(motion) < 40 or len(motion) >= 200:
            continue
        if not np.isfinite(motion).all():
            continue
        motion = motion.astype(np.float64, copy=False)
        total_frames += len(motion)
        motion_sum += motion.sum(axis=0)
        motion_square_sum += np.square(motion).sum(axis=0)

    if total_frames == 0:
        raise ValueError(
            f"No valid raw joint motions found in {motion_dir} for {split_file}."
        )

    mean = motion_sum / total_frames
    variance = np.maximum(
        motion_square_sum / total_frames - np.square(mean),
        0.0,
    )
    std = np.sqrt(variance)
    return _validate_raw_joint_mean_std(
        mean,
        std,
        "per_coordinate",
        joint_count,
        joint_dim,
    )


def load_raw_joint_mean_std(
    dataset_name,
    data_root=None,
    mean_path=None,
    std_path=None,
    normalization="shared_xyz",
    joint_count=22,
    joint_dim=3,
    split_file=None,
):
    dataset_name = str(dataset_name)
    normalization = normalize_raw_joint_normalization(normalization)
    candidates = [(mean_path, std_path)]
    if normalization == "per_coordinate":
        candidates.extend(
            [
                (
                    f"utils/22x3_mean_std/{dataset_name}/"
                    "22x3_per_coordinate_mean.npy",
                    f"utils/22x3_mean_std/{dataset_name}/"
                    "22x3_per_coordinate_std.npy",
                ),
                (
                    f"ProjFlow/utils/22x3_mean_std/{dataset_name}/"
                    "22x3_per_coordinate_mean.npy",
                    f"ProjFlow/utils/22x3_mean_std/{dataset_name}/"
                    "22x3_per_coordinate_std.npy",
                ),
            ]
        )
        if data_root is not None:
            candidates.append(
                (
                    Path(data_root) / "Mean_22x3_per_coordinate.npy",
                    Path(data_root) / "Std_22x3_per_coordinate.npy",
                )
            )
    else:
        candidates.extend(
            [
                (
                    f"utils/22x3_mean_std/{dataset_name}/22x3_mean.npy",
                    f"utils/22x3_mean_std/{dataset_name}/22x3_std.npy",
                ),
                (
                    f"ProjFlow/utils/22x3_mean_std/{dataset_name}/22x3_mean.npy",
                    f"ProjFlow/utils/22x3_mean_std/{dataset_name}/22x3_std.npy",
                ),
            ]
        )
        if data_root is not None:
            candidates.append(
                (
                    Path(data_root) / "Mean_22x3.npy",
                    Path(data_root) / "Std_22x3.npy",
                )
            )

    mean_file, std_file = _first_existing_pair(candidates)
    if mean_file is not None and std_file is not None:
        return _validate_raw_joint_mean_std(
            np.load(mean_file),
            np.load(std_file),
            normalization,
            joint_count,
            joint_dim,
        )

    if normalization == "per_coordinate" and data_root is not None:
        motion_dir = Path(data_root) / "new_joints"
        if split_file is None:
            split_file = Path(data_root) / "train.txt"
        print(f"Computing per-coordinate raw joint mean/std from {split_file}.")
        return compute_raw_joint_mean_std(
            motion_dir,
            split_file,
            joint_count=joint_count,
            joint_dim=joint_dim,
        )

    if normalization == "per_coordinate":
        expected_files = "22x3_per_coordinate_mean.npy and 22x3_per_coordinate_std.npy"
    else:
        expected_files = "22x3_mean.npy and 22x3_std.npy"
    raise FileNotFoundError(
        f"Could not find raw {normalization} mean/std files ({expected_files}). "
        "Set raw_mean_path and raw_std_path or provide data_root so "
        "per-coordinate statistics can be computed from the training split."
    )


def flatten_raw_joints(motion, joint_count=22, joint_dim=3):
    if motion.dim() != 4:
        raise ValueError(
            "Raw joint motion must have shape [B, T, J, D] or [B, T, 1, J*D], "
            f"got {tuple(motion.shape)}."
        )

    flat_dim = joint_count * joint_dim
    if motion.shape[2] == 1 and motion.shape[3] == flat_dim:
        return motion
    if motion.shape[2] != joint_count or motion.shape[3] != joint_dim:
        raise ValueError(
            f"Expected raw joint shape [B, T, {joint_count}, {joint_dim}], "
            f"got {tuple(motion.shape)}."
        )
    return motion.reshape(motion.shape[0], motion.shape[1], 1, flat_dim)


def unflatten_raw_joints_np(motion, joint_count=22, joint_dim=3):
    flat_dim = joint_count * joint_dim
    if (
        motion.ndim == 4
        and motion.shape[2] == joint_count
        and motion.shape[3] == joint_dim
    ):
        return motion
    if motion.ndim == 4 and motion.shape[2] == 1 and motion.shape[3] == flat_dim:
        return motion.reshape(motion.shape[0], motion.shape[1], joint_count, joint_dim)
    if motion.ndim == 3 and motion.shape[-1] == flat_dim:
        return motion.reshape(motion.shape[0], motion.shape[1], joint_count, joint_dim)
    raise ValueError(
        "Raw joint motion must have shape [B, T, J, D], [B, T, 1, J*D], "
        f"or [B, T, J*D], got {tuple(motion.shape)}."
    )


def as_numpy(motion):
    if torch.is_tensor(motion):
        return motion.detach().cpu().numpy()
    return np.asarray(motion)
