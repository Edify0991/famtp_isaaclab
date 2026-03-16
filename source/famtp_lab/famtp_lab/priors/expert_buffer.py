"""Expert motion buffer loading from no-transition dataset index."""

from __future__ import annotations

from dataclasses import dataclass
import random

from famtp_lab.motion.dataset_index import DatasetIndex


@dataclass
class ExpertSegment:
    """Within-skill segment metadata used by discriminator training."""

    clip_path: str
    skill_label: str
    start_time: float
    end_time: float


class ExpertBuffer:
    """Lightweight expert segment store backed by dataset index."""

    def __init__(self, index_path: str):
        index = DatasetIndex.from_json(index_path)
        self._segments = [
            ExpertSegment(
                clip_path=e.clip_path,
                skill_label=e.skill_label,
                start_time=e.start_time,
                end_time=e.end_time,
            )
            for e in index.entries
        ]

    def __len__(self) -> int:
        return len(self._segments)

    def sample_batch(self, batch_size: int) -> list[ExpertSegment]:
        """Return list[ExpertSegment] of length batch_size (with replacement)."""
        if not self._segments:
            raise ValueError("ExpertBuffer is empty.")
        return [random.choice(self._segments) for _ in range(batch_size)]

    def skill_counts(self) -> dict[str, int]:
        """Count segments by skill label."""
        counts: dict[str, int] = {}
        for seg in self._segments:
            counts[seg.skill_label] = counts.get(seg.skill_label, 0) + 1
        return counts
