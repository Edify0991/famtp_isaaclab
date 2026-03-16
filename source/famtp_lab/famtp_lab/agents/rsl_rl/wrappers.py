"""Environment wrappers for RSL-RL integration."""

from __future__ import annotations

import gymnasium as gym


class RslRlPolicyObsWrapper(gym.ObservationWrapper):
    """Expose the ``policy`` observation tensor expected by RSL-RL."""

    def observation(self, observation):
        if isinstance(observation, dict) and "policy" in observation:
            return observation["policy"]
        return observation
