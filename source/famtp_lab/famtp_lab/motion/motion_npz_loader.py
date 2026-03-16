"""Robust NPZ motion loader and canonicalization for G1 motion clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema_inspector import infer_fps_from_name


def _as_array(value) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _find_key(data: dict[str, np.ndarray], candidates: list[str]) -> str | None:
    for cand in candidates:
        for key in data:
            if cand in key.lower():
                return key
    return None


def load_npz_canonical(path: str | Path) -> dict:
    """Load NPZ file and normalize into canonical clip dictionary.

    Canonical keys:
        fps, num_frames, root_pos, root_quat, joint_pos,
        joint_vel, body_pos, body_quat, source_file, parser_notes
    """
    path = Path(path)
    parser_notes = []
    with np.load(path, allow_pickle=True) as npz:
        raw = {k: _as_array(npz[k]) for k in npz.keys()}

    # Drop invalid keys.
    raw = {k: v for k, v in raw.items() if v is not None}

    fps = None
    for key in raw:
        if key.lower() in {"fps", "framerate", "frame_rate"}:
            try:
                fps = int(np.asarray(raw[key]).item())
                parser_notes.append(f"fps from key={key}")
            except Exception:
                fps = None
            break
    if fps is None:
        fps = infer_fps_from_name(path.as_posix()) or 120
        parser_notes.append("fps inferred from name or default=120")

    joint_key = _find_key(raw, ["jpos", "joint_pos", "qpos", "dof_pos", "pose"])
    root_pos_key = _find_key(raw, ["root_pos", "trans", "root_translation", "pelvis_pos"])
    root_quat_key = _find_key(raw, ["root_quat", "root_rot", "quat", "orientation"])
    body_pos_key = _find_key(raw, ["body_pos", "global_pos", "link_pos"])
    body_quat_key = _find_key(raw, ["body_quat", "link_quat", "global_quat"])
    joint_vel_key = _find_key(raw, ["joint_vel", "qvel", "dof_vel"])

    joint_pos = raw.get(joint_key) if joint_key else None
    root_pos = raw.get(root_pos_key) if root_pos_key else None
    root_quat = raw.get(root_quat_key) if root_quat_key else None
    body_pos = raw.get(body_pos_key) if body_pos_key else None
    body_quat = raw.get(body_quat_key) if body_quat_key else None
    joint_vel = raw.get(joint_vel_key) if joint_vel_key else None

    # Frame count fallback.
    candidates = [v.shape[0] for v in [joint_pos, root_pos, root_quat, body_pos, body_quat, joint_vel] if isinstance(v, np.ndarray) and v.ndim > 0]
    num_frames = int(max(candidates)) if candidates else 0

    if joint_pos is None:
        parser_notes.append("joint_pos not found; creating zeros fallback")
        joint_pos = np.zeros((num_frames, 0), dtype=np.float32)
    if root_pos is None:
        parser_notes.append("root_pos not found; using zeros")
        root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    if root_quat is None:
        parser_notes.append("root_quat not found; using identity quaternion")
        root_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_frames, 1))

    return {
        "fps": int(fps),
        "num_frames": int(num_frames),
        "root_pos": root_pos,
        "root_quat": root_quat,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos": body_pos,
        "body_quat": body_quat,
        "source_file": path.as_posix(),
        "parser_notes": parser_notes,
    }
