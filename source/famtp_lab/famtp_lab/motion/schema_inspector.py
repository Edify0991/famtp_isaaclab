"""Schema inspection utilities for G1-retargeted motion files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import numpy as np


FPS_PATTERNS = [r"(?:_|-)(\d{2,4})fps", r"(?:_|-)(\d{2,4})(?:_|-)hz", r"poses_(\d{2,4})"]


def infer_fps_from_name(path: str) -> int | None:
    """Infer FPS from file name pattern, e.g. ``*_poses_120_jpos.npz``."""
    name = Path(path).name.lower()
    for pat in FPS_PATTERNS:
        match = re.search(pat, name)
        if match:
            value = int(match.group(1))
            if 20 <= value <= 480:
                return value
    return None


def _safe_shape(value) -> list[int]:
    if hasattr(value, "shape"):
        return [int(v) for v in value.shape]
    return []


def inspect_npz_file(path: Path) -> dict:
    """Inspect one NPZ file and return key metadata.

    The function is robust to unknown key names and mixed scalar/array content.
    """
    with np.load(path, allow_pickle=True) as npz:
        keys = list(npz.keys())
        key_meta = {}
        candidate_frames = []
        for key in keys:
            value = npz[key]
            shape = _safe_shape(value)
            if shape:
                candidate_frames.append(shape[0])
            key_meta[key] = {
                "shape": shape,
                "dtype": str(getattr(value, "dtype", type(value).__name__)),
            }

        fps = None
        for key in keys:
            low = key.lower()
            if low in {"fps", "framerate", "frame_rate"}:
                try:
                    fps = int(np.asarray(npz[key]).item())
                except Exception:
                    fps = None
                break
        if fps is None:
            fps = infer_fps_from_name(path.as_posix())

        num_frames = max(candidate_frames) if candidate_frames else None
        duration_s = (num_frames / fps) if (num_frames and fps) else None

        keyset_sig = tuple(sorted(keys))
        return {
            "path": path.as_posix(),
            "suffix": path.suffix,
            "keys": key_meta,
            "num_frames": num_frames,
            "fps": fps,
            "duration_s": duration_s,
            "keyset_signature": keyset_sig,
        }


def infer_field_candidates(key_names: list[str]) -> dict[str, list[str]]:
    """Infer likely semantic fields from key names only (no hard assumption)."""
    lower = [k.lower() for k in key_names]

    def pick(patterns: list[str]) -> list[str]:
        out = []
        for raw in key_names:
            low = raw.lower()
            if any(p in low for p in patterns):
                out.append(raw)
        return out

    return {
        "joint_pos_candidates": pick(["jpos", "joint", "qpos", "dof_pos", "pose"]),
        "root_pos_candidates": pick(["root_pos", "trans", "root_translation", "pelvis_pos"]),
        "root_quat_candidates": pick(["root_quat", "quat", "root_rot", "orientation"]),
        "body_pos_candidates": pick(["body_pos", "global_pos", "link_pos"]),
        "body_quat_candidates": pick(["body_quat", "link_quat", "global_quat"]),
    }


def summarize_scan(file_reports: list[dict]) -> dict:
    """Build aggregate schema statistics across scan reports."""
    key_counter: Counter[str] = Counter()
    keyset_counter: Counter[tuple[str, ...]] = Counter()
    suffix_counter: Counter[str] = Counter()
    fps_counter: Counter[int] = Counter()
    file_pattern_counter: Counter[str] = Counter()

    all_keys: list[str] = []
    for rep in file_reports:
        suffix_counter[rep["suffix"]] += 1
        if rep.get("fps") is not None:
            fps_counter[int(rep["fps"])] += 1
        keyset = tuple(rep.get("keyset_signature", ()))
        keyset_counter[keyset] += 1
        for key in rep.get("keys", {}).keys():
            key_counter[key] += 1
            all_keys.append(key)

        name = Path(rep["path"]).name
        file_pattern_counter[re.sub(r"\d+", "<num>", name)] += 1

    field_candidates = infer_field_candidates(sorted(set(all_keys)))

    return {
        "num_files": len(file_reports),
        "suffix_counts": dict(suffix_counter.most_common()),
        "fps_counts": dict(fps_counter.most_common()),
        "key_frequency": dict(key_counter.most_common()),
        "keyset_combinations": [
            {"keys": list(combo), "count": count} for combo, count in keyset_counter.most_common(20)
        ],
        "file_name_patterns": [
            {"pattern": pat, "count": cnt} for pat, cnt in file_pattern_counter.most_common(20)
        ],
        "field_candidates": field_candidates,
    }
