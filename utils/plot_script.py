import os
from textwrap import wrap
from typing import Optional

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.motion_process import t2m_kinematic_chain

skeleton = t2m_kinematic_chain


def plot_3d_motion(
    save_path: str,
    joints: np.ndarray,
    title: str,
    figsize: tuple[int, int] = (3, 3),
    fps: int = 120,
    radius: int = 3,
    kinematic_tree: list = skeleton,
    hint: Optional[np.ndarray] = None,
    highlight_pelvis: bool = False,
    hint_colors: Optional[np.ndarray] = None,
) -> None:

    title = "\n".join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    data *= 1.3  # scale for visualization
    if hint is not None:
        mask = hint.sum(-1) != 0
        hint = hint[mask]
        if hint_colors is not None:
            hint_colors = np.asarray(hint_colors)[mask]
        hint *= 1.3

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = [
        "#DD5A37",
        "#D69E00",
        "#B75A39",
        "#DD5A37",
        "#D69E00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
    ]

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    if hint is not None:
        hint[..., 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )

        if hint is not None:
            ax.scatter(
                hint[..., 0] - trajec[index, 0],
                hint[..., 1],
                hint[..., 2] - trajec[index, 1],
                color=hint_colors if hint_colors is not None else "#80B79A",
            )

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        # Highlight pelvis joint (joint 0) if requested
        if highlight_pelvis:
            pelvis_pos = data[index, 0]  # Get pelvis position (joint 0)
            # Plot a large red sphere for the pelvis
            ax.scatter(
                pelvis_pos[0],
                pelvis_pos[1],
                pelvis_pos[2],
                color="red",
                s=200,
                alpha=0.8,
                edgecolors="darkred",
                linewidth=2,
            )

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )
    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion_long(
    save_path: str,
    joints: np.ndarray,
    texts: list[str],
    lengths: list[int],
    figsize: tuple[int, int] = (3, 3),
    fps: int = 120,
    radius: int = 3,
    kinematic_tree: list = skeleton,
    hint: Optional[np.ndarray] = None,
    highlight_pelvis: bool = False,
) -> None:
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    data *= 1.3  # scale for visualization
    if hint is not None:
        mask = hint.sum(-1) != 0
        hint = hint[mask]
        hint *= 1.3

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = [
        "#DD5A37",
        "#D69E00",
        "#B75A39",
        "#DD5A37",
        "#D69E00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#FF6D00",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
        "#DDB50E",
    ]

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    if hint is not None:
        hint[..., 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    lengths_cumsum = np.cumsum(lengths)

    suptitle = fig.suptitle(
        "",
        x=0.5,
        y=0.98,
        ha="center",
        va="top",
        fontsize=10,
        bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
    )

    def update(index):
        # Determine current segment by frame index and update dynamic title
        seg_idx = int(np.searchsorted(lengths_cumsum, index, side="right"))
        text = "\n".join(wrap(texts[seg_idx], 20))
        suptitle.set_text(text)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )

        if hint is not None:
            ax.scatter(
                hint[..., 0] - trajec[index, 0],
                hint[..., 1],
                hint[..., 2] - trajec[index, 1],
                color="#80B79A",
            )

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        # Highlight pelvis joint (joint 0) if requested
        if highlight_pelvis:
            pelvis_pos = data[index, 0]  # Get pelvis position (joint 0)
            # Plot a large red sphere for the pelvis
            ax.scatter(
                pelvis_pos[0],
                pelvis_pos[1],
                pelvis_pos[2],
                color="red",
                s=200,
                alpha=0.8,
                edgecolors="darkred",
                linewidth=2,
            )

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )
    ani.save(save_path, fps=fps)
    plt.close()


def save_multiple_samples(
    args,
    out_path,
    row_print_template,
    all_print_template,
    row_file_template,
    all_file_template,
    caption,
    num_samples_in_out_file,
    rep_files,
    sample_files,
    sample_i,
):

    sample_files.append(rep_files[0])

    # stack horizontally if there are multiple repetitions
    if sample_i + 1 == num_samples_in_out_file:
        # if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_repetitions:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(
            sample_i - len(sample_files) + 1, sample_i
        )
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(
            all_print_template.format(
                sample_i - len(sample_files) + 1, sample_i, all_sample_save_file
            )
        )
        ffmpeg_rep_files = [f" -i {f} " for f in sample_files]
        vstack_args = (
            f" -filter_complex hstack=inputs={len(sample_files)}"
            if len(sample_files) > 1
            else ""
        )
        ffmpeg_rep_cmd = (
            f"ffmpeg -y -loglevel warning "
            + "".join(ffmpeg_rep_files)
            + f"{vstack_args} {all_sample_save_path}"
        )
        os.system(ffmpeg_rep_cmd)
        os.system(
            f"ffmpeg -y -loglevel warning -i {all_sample_save_path} {all_sample_save_path.replace('.mp4', '.gif')}"
        )
        sample_files = []
    return sample_files
