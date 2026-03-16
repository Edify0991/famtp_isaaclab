"""Dataset index structures for motion clips used in week-1 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class ClipEntry:
    """Single motion clip metadata entry.

    Times are in seconds in source motion timeline.
    """

    clip_path: str
    skill_label: str
    start_time: float
    end_time: float
    segment_id: str | None = None

    @property
    def duration(self) -> float:
        """Clip duration in seconds."""
        return max(0.0, self.end_time - self.start_time)


@dataclass
class DatasetIndex:
    """Container for motion clip metadata entries."""

    entries: list[ClipEntry]

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetIndex":
        """Load index from JSON file."""
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            raw_entries = data.get("entries", [])
        else:
            raw_entries = data
        return cls(entries=[ClipEntry(**entry) for entry in raw_entries])

    def to_json(self, path: str | Path) -> None:
        """Serialize index to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([asdict(entry) for entry in self.entries], indent=2))

    def skill_duration(self) -> dict[str, float]:
        """Aggregate total duration per skill."""
        durations: dict[str, float] = {}
        for entry in self.entries:
            durations[entry.skill_label] = durations.get(entry.skill_label, 0.0) + entry.duration
        return durations
