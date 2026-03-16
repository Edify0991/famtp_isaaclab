"""Switch-window metric computation for week-1 feasibility experiments."""

from __future__ import annotations

import numpy as np


def skill_switch_accuracy(current_skill, target_skill):
    """Compute fraction of environments where current equals target skill."""
    return (current_skill == target_skill).float().mean()


def _integral(values: np.ndarray, dt: float) -> float:
    return float(np.sum(np.abs(values)) * dt)


def _peak(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def compute_switch_window_metrics(
    joint_jerk_norm: np.ndarray,
    com_jerk_norm: np.ndarray,
    foot_slip_norm: np.ndarray,
    ground_penetration_depth: np.ndarray,
    torque_norm: np.ndarray,
    torque_rate_norm: np.ndarray,
    root_pitch: np.ndarray,
    root_roll: np.ndarray,
    alive_mask: np.ndarray,
    switch_success_mask: np.ndarray,
    switch_event_indices: np.ndarray,
    dt: float,
    window_s: float = 0.75,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Compute aggregate and per-switch metrics.

    Arrays are episode time-series with shape ``(T,)``.
    """
    total_steps = len(alive_mask)
    survival_time = float(np.sum(alive_mask) * dt)
    fall_rate = float(1.0 - np.mean(alive_mask))
    mean_survival_time = survival_time

    root_dev = np.sqrt(np.square(root_pitch) + np.square(root_roll))

    metrics = {
        "fall_rate": fall_rate,
        "mean_survival_time": mean_survival_time,
        "joint_jerk_peak": _peak(joint_jerk_norm),
        "joint_jerk_integral": _integral(joint_jerk_norm, dt),
        "com_jerk_integral": _integral(com_jerk_norm, dt),
        "foot_slip_integral_during_contact": _integral(foot_slip_norm, dt),
        "ground_penetration_count": float(np.sum(ground_penetration_depth > 1e-4)),
        "torque_peak": _peak(torque_norm),
        "torque_rate_integral": _integral(torque_rate_norm, dt),
        "switch_success_rate": float(np.mean(switch_success_mask)) if switch_success_mask.size else 1.0,
        "root_pitch_roll_deviation": _integral(root_dev, dt),
    }

    half_window_steps = int(round(window_s / dt))
    per_switch: list[dict[str, float]] = []
    for switch_idx in switch_event_indices.tolist():
        start = max(0, switch_idx - half_window_steps)
        end = min(total_steps, switch_idx + half_window_steps + 1)
        per_switch.append(
            {
                "switch_step": float(switch_idx),
                "window_start_step": float(start),
                "window_end_step": float(end - 1),
                "time_since_last_switch_s": float((switch_idx - per_switch[-1]["switch_step"]) * dt) if per_switch else 0.0,
                "joint_jerk_integral": _integral(joint_jerk_norm[start:end], dt),
                "com_jerk_integral": _integral(com_jerk_norm[start:end], dt),
                "torque_rate_integral": _integral(torque_rate_norm[start:end], dt),
                "switch_success": float(switch_success_mask[min(switch_idx, len(switch_success_mask) - 1)]),
            }
        )

    return metrics, per_switch
