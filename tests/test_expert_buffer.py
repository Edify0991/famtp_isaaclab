"""Tests for no-transition expert buffer loading/sampling."""

import json
from pathlib import Path

from famtp_lab.priors.expert_buffer import ExpertBuffer


def test_expert_buffer_load_and_sample(tmp_path: Path) -> None:
    index_path = tmp_path / "no_transition_motion_index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "clip_path": "clips/a.npy",
                    "skill_label": "locomotion_run",
                    "start_time": 0.2,
                    "end_time": 1.7,
                    "segment_id": "a",
                },
                {
                    "clip_path": "clips/b.npy",
                    "skill_label": "arm_dribble_like",
                    "start_time": 0.1,
                    "end_time": 2.4,
                    "segment_id": "b",
                },
            ]
        )
    )

    buf = ExpertBuffer(str(index_path))
    assert len(buf) == 2
    batch = buf.sample_batch(5)
    assert len(batch) == 5
    counts = buf.skill_counts()
    assert counts["locomotion_run"] == 1
    assert counts["arm_dribble_like"] == 1
