"""Sampling helpers for within-skill motion clips."""

from __future__ import annotations

import random

from .dataset_index import ClipEntry, DatasetIndex


class SkillClipSampler:
    """Simple skill-conditioned clip sampler.

    This utility is intentionally small and is a hook point for future
    AMP/FaMTP curriculum sampling.
    """

    def __init__(self, dataset_index: DatasetIndex):
        self._by_skill: dict[str, list[ClipEntry]] = {}
        for entry in dataset_index.entries:
            self._by_skill.setdefault(entry.skill_label, []).append(entry)

    def sample(self, skill_label: str) -> ClipEntry:
        """Sample one clip for a requested skill."""
        if skill_label not in self._by_skill or not self._by_skill[skill_label]:
            raise ValueError(f"No clips available for skill '{skill_label}'")
        return random.choice(self._by_skill[skill_label])

    def available_skills(self) -> list[str]:
        """Return sorted list of skill labels in dataset."""
        return sorted(self._by_skill.keys())
