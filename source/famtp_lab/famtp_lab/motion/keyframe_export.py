"""Export keyframe strips, root trajectories, and joint curves for G1 clips."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def export_keyframe_strip(clip: dict, output_path: str | Path, num_keyframes: int = 8) -> str:
    """Export a simple keyframe stripe using joint-position heat snapshots."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joint_pos = np.asarray(clip.get("joint_pos"))
    num_frames = int(clip.get("num_frames", len(joint_pos)))
    if num_frames <= 0:
        raise ValueError("Clip has zero frames.")

    idx = np.linspace(0, num_frames - 1, num=min(num_keyframes, num_frames), dtype=int)
    if joint_pos.ndim == 2 and joint_pos.shape[1] > 0:
        stripe = joint_pos[idx, :]
    else:
        stripe = np.zeros((len(idx), 1), dtype=np.float32)

    plt.figure(figsize=(10, 2.5))
    plt.imshow(stripe.T, aspect="auto", cmap="viridis")
    plt.title(f"Keyframe strip: {Path(clip.get('source_file','clip')).name}")
    plt.xlabel("keyframe index")
    plt.ylabel("joint dim")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path.as_posix()


def export_root_trajectory_plot(clip: dict, output_path: str | Path) -> str:
    """Export root XY trajectory plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root_pos = np.asarray(clip.get("root_pos"))
    if root_pos.ndim != 2 or root_pos.shape[1] < 2:
        root_pos = np.zeros((max(1, int(clip.get("num_frames", 1))), 3), dtype=np.float32)

    plt.figure(figsize=(5, 5))
    plt.plot(root_pos[:, 0], root_pos[:, 1])
    plt.scatter(root_pos[0, 0], root_pos[0, 1], c="green", label="start")
    plt.scatter(root_pos[-1, 0], root_pos[-1, 1], c="red", label="end")
    plt.title("Root XY trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path.as_posix()


def export_joint_curves(clip: dict, output_path: str | Path, max_joints: int = 12) -> str:
    """Export joint angle time curves for first ``max_joints`` channels."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jpos = np.asarray(clip.get("joint_pos"))
    if jpos.ndim != 2 or jpos.shape[1] == 0:
        jpos = np.zeros((max(1, int(clip.get("num_frames", 1))), 1), dtype=np.float32)

    keep = min(max_joints, jpos.shape[1])
    plt.figure(figsize=(9, 4))
    for i in range(keep):
        plt.plot(jpos[:, i], linewidth=0.8)
    plt.title("Joint position curves")
    plt.xlabel("frame")
    plt.ylabel("joint angle")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path.as_posix()
