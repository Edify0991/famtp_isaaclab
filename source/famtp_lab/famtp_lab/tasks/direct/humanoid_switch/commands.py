"""Command interface for skill-switching experiments."""

from dataclasses import dataclass

import torch


@dataclass
class SkillCommandGenerator:
    """Simple static command generator.

    The initial version uses fixed target skill ids for all environments.
    """

    num_skills: int = 4
    default_target_skill: int = 0

    def sample(self, num_envs: int, device: torch.device | str) -> torch.Tensor:
        """Return target skill IDs with shape ``(num_envs,)``."""
        target = torch.full((num_envs,), self.default_target_skill, device=device, dtype=torch.long)
        return torch.clamp(target, min=0, max=max(self.num_skills - 1, 0))
