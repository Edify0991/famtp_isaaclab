"""Gymnasium registration utilities for FaMTP tasks."""

from __future__ import annotations

import gymnasium as gym


def registered_famtp_tasks(prefix: str = "FaMTP-") -> list[str]:
    """Return all registered task ids for this project."""
    return sorted(spec.id for spec in gym.registry.values() if spec.id.startswith(prefix))
