"""Motion dataset loading and no-transition filtering utilities."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from .dataset_index import ClipEntry, DatasetIndex


DEFAULT_SKILLS = ["locomotion_run", "arm_dribble_like", "stop_shoot_pose"]


def filter_no_transition_segments(
    dataset_index: DatasetIndex,
    boundary_window_s: float,
    allowed_skills: list[str] | None = None,
) -> tuple[DatasetIndex, dict]:
    """Filter clips to within-skill portions by trimming boundary windows.

    A boundary window is removed from start and end of each segment, ensuring
    remaining clips are less likely to include transition dynamics.
    """
    allowed = set(allowed_skills or DEFAULT_SKILLS)
    kept: list[ClipEntry] = []
    removed_count = 0
    removed_duration = 0.0
    removed_by_skill: dict[str, float] = {}

    for entry in dataset_index.entries:
        if entry.skill_label not in allowed:
            removed_count += 1
            removed_duration += entry.duration
            removed_by_skill[entry.skill_label] = removed_by_skill.get(entry.skill_label, 0.0) + entry.duration
            continue

        trimmed_start = entry.start_time + boundary_window_s
        trimmed_end = entry.end_time - boundary_window_s
        if trimmed_end <= trimmed_start:
            removed_count += 1
            removed_duration += entry.duration
            removed_by_skill[entry.skill_label] = removed_by_skill.get(entry.skill_label, 0.0) + entry.duration
            continue

        kept.append(
            ClipEntry(
                clip_path=entry.clip_path,
                skill_label=entry.skill_label,
                start_time=trimmed_start,
                end_time=trimmed_end,
                segment_id=entry.segment_id,
            )
        )

    filtered = DatasetIndex(entries=kept)
    summary = {
        "total_clips_before": len(dataset_index.entries),
        "total_clips_after": len(filtered.entries),
        "per_skill_duration_before": dataset_index.skill_duration(),
        "per_skill_duration_after": filtered.skill_duration(),
        "removed_boundary_statistics": {
            "removed_clip_count": removed_count,
            "removed_duration_s": removed_duration,
            "removed_duration_by_skill_s": removed_by_skill,
            "boundary_window_s": boundary_window_s,
        },
    }
    return filtered, summary


def build_no_transition_dataset(
    input_index_path: str | Path,
    output_index_path: str | Path,
    summary_path: str | Path,
    boundary_window_s: float,
    allowed_skills: list[str] | None = None,
) -> dict:
    """Load, filter, and write no-transition dataset artifacts."""
    dataset = DatasetIndex.from_json(input_index_path)
    filtered, summary = filter_no_transition_segments(dataset, boundary_window_s, allowed_skills=allowed_skills)
    filtered.to_json(output_index_path)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    return summary
