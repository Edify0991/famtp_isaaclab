"""Tests for G1 schema inspection and canonical loader."""

import numpy as np
from pathlib import Path

from famtp_lab.motion.motion_npz_loader import load_npz_canonical
from famtp_lab.motion.schema_inspector import inspect_npz_file, summarize_scan


def test_inspect_and_load_npz(tmp_path: Path) -> None:
    path = tmp_path / "demo_poses_120_jpos.npz"
    np.savez(
        path,
        jpos=np.zeros((20, 29), dtype=np.float32),
        root_pos=np.zeros((20, 3), dtype=np.float32),
        root_quat=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (20, 1)),
    )

    rep = inspect_npz_file(path)
    assert rep["num_frames"] == 20
    assert rep["fps"] == 120

    clip = load_npz_canonical(path)
    assert clip["num_frames"] == 20
    assert clip["fps"] == 120
    assert clip["joint_pos"].shape == (20, 29)

    summary = summarize_scan([rep])
    assert summary["num_files"] == 1
