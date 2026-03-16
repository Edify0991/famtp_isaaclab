"""Command and skill-switch protocol for humanoid-switch experiments."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch

SKILL_LABELS = ["locomotion_run", "arm_dribble_like", "stop_shoot_pose"]
SKILL_TO_ID = {name: idx for idx, name in enumerate(SKILL_LABELS)}
FIXED_CHAIN = [SKILL_TO_ID["locomotion_run"], SKILL_TO_ID["arm_dribble_like"], SKILL_TO_ID["stop_shoot_pose"]]


@dataclass
class SkillSwitchScheduler:
    """Switch timing and target selection.

    Supports fixed-chain and random-chain modes for week-1 feasibility runs.
    """

    switch_time_min_s: float = 1.0
    switch_time_max_s: float = 3.0
    chain_mode: str = "fixed"  # fixed|random
    num_skills: int = len(SKILL_LABELS)

    def sample_next_switch_step(self, sim_dt: float, decimation: int) -> int:
        """Sample next switch step count in env-step units."""
        switch_time_s = random.uniform(self.switch_time_min_s, self.switch_time_max_s)
        env_dt = sim_dt * decimation
        return max(1, int(round(switch_time_s / env_dt)))

    def next_target(self, current_skill_id: int) -> int:
        """Return the next target skill id based on chain mode."""
        if self.chain_mode == "fixed":
            if current_skill_id not in FIXED_CHAIN:
                return FIXED_CHAIN[0]
            index = FIXED_CHAIN.index(current_skill_id)
            return FIXED_CHAIN[(index + 1) % len(FIXED_CHAIN)]

        if self.chain_mode == "random":
            candidates = [idx for idx in range(self.num_skills) if idx != current_skill_id]
            return random.choice(candidates) if candidates else current_skill_id

        raise ValueError(f"Unsupported chain_mode '{self.chain_mode}'")


def skill_id_tensor_to_labels(skill_ids: torch.Tensor) -> list[str]:
    """Convert a tensor of skill ids to label strings."""
    labels: list[str] = []
    for idx in skill_ids.tolist():
        labels.append(SKILL_LABELS[idx] if 0 <= idx < len(SKILL_LABELS) else "unknown")
    return labels
