import numpy as np
import torch

from utils.motion_process import (
    kit_kinematic_chain,
    kit_raw_offsets,
    t2m_kinematic_chain,
    t2m_raw_offsets,
)
from utils.quaternion import (
    qfix,
    qinv_np,
    qmul_np,
    qrot_np,
    quaternion_to_cont6d_np,
)
from utils.skeleton import Skeleton


_CONFIGS = {
    "t2m": {
        "raw_offsets": torch.from_numpy(t2m_raw_offsets),
        "kinematic_chain": t2m_kinematic_chain,
        "face_joint_indx": [2, 1, 17, 16],
        "fid_r": [8, 11],
        "fid_l": [7, 10],
    },
    "kit": {
        "raw_offsets": torch.from_numpy(kit_raw_offsets),
        "kinematic_chain": kit_kinematic_chain,
        "face_joint_indx": [11, 16, 5, 8],
        "fid_r": [14, 15],
        "fid_l": [19, 20],
    },
}


def _select_config(data, dataset_name=None):
    if dataset_name is not None:
        key = dataset_name.lower()
        if key in ("kit", "kit-ml", "kit_ml"):
            return _CONFIGS["kit"]
        if key in ("t2m", "humanml3d", "humanml"):
            return _CONFIGS["t2m"]
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    joints = int(data.shape[1])
    if joints == 21:
        return _CONFIGS["kit"]
    if joints == 22:
        return _CONFIGS["t2m"]
    raise ValueError(f"Unsupported joints count: {joints}")


def process_file(positions, feet_thre, config):
    positions = np.asarray(positions).copy()
    n_raw_offsets = config["raw_offsets"]
    kinematic_chain = config["kinematic_chain"]
    face_joint_indx = config["face_joint_indx"]
    fid_r, fid_l = config["fid_r"], config["fid_l"]
    global_positions = positions.copy()

    def foot_detect(positions, thres):
        velfactor = np.array([thres, thres])
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    def get_rifke(positions, r_rot):
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        return qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(
            positions,
            face_joint_indx,
            smooth_forward=True,
        )
        quat_params = qfix(quat_params)
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions, r_rot)

    root_y = positions[:, 0, 1:2]
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    return data, global_positions, positions, l_velocity


def back_process(data, dataset_name=None):
    config = _select_config(data, dataset_name=dataset_name)
    data, _, _, _ = process_file(data, 0.002, config)
    joints_num = int(config["raw_offsets"].shape[0])
    pose_dim = 4 + (joints_num - 1) * 3
    return data[:, :pose_dim]


def back_process_full(data, dataset_name=None):
    config = _select_config(data, dataset_name=dataset_name)
    full_data, _, _, _ = process_file(data, 0.002, config)
    return full_data
