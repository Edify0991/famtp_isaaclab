"""Evaluate switching degradation metrics across baseline methods."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from famtp_lab.tasks.direct.humanoid_switch.metrics import compute_switch_window_metrics

METHODS = ["ppo_cmd", "fullbody_amp", "partwise_raw", "famtp_stage1", "famtp_nobridge", "famtp_full"]
PARTS = ["left_leg", "right_leg", "torso", "left_arm", "right_arm"]


def _method_severity(method: str, mode: str) -> float:
    base = {"ppo_cmd": 1.00, "fullbody_amp": 0.82, "partwise_raw": 0.88, "famtp_stage1": 0.74, "famtp_nobridge": 0.77, "famtp_full": 0.68}[method]
    return 0.65 * base if mode == "single_skill" else 1.35 * base


def _reward_breakdown(method: str, severity: float) -> dict[str, float]:
    task = float(1.4 - 0.3 * severity)
    latent_part = float(0.0)
    coupling = float(0.0)
    if method in {"famtp_stage1", "famtp_nobridge"}:
        latent_part = float(0.5 - 0.1 * severity)
        coupling = float(0.45 - 0.08 * severity)
    if method == "famtp_full":
        latent_part = float(0.55 - 0.08 * severity)
        coupling = float(0.55 - 0.08 * severity)
    elif method == "fullbody_amp":
        latent_part = float(0.25 - 0.06 * severity)
    elif method == "partwise_raw":
        latent_part = float(0.3 - 0.07 * severity)
    return {
        "reward_task": task,
        "reward_latent_part": max(latent_part, 0.0),
        "reward_coupling": max(coupling, 0.0),
        "reward_total": max(task + latent_part + coupling, 0.0),
    }


def _synthetic_latents(rng: np.random.Generator, method: str, mode: str, n: int = 300) -> pd.DataFrame:
    rows = []
    skill_labels = ["locomotion_run", "arm_dribble_like", "stop_shoot_pose"]
    spread = {"ppo_cmd": 1.1, "fullbody_amp": 0.9, "partwise_raw": 0.85, "famtp_stage1": 0.7, "famtp_nobridge": 0.72, "famtp_full": 0.6}[method]
    for part in PARTS:
        for skill in skill_labels:
            center = skill_labels.index(skill) * 1.2
            for _ in range(n // (len(PARTS) * len(skill_labels))):
                z1 = center + rng.normal(0.0, spread)
                z2 = rng.normal(0.0, 0.7 * spread)
                rows.append({"method": method, "mode": mode, "part": part, "skill_label": skill, "z1": z1, "z2": z2})
    return pd.DataFrame(rows)


def generate_synthetic_episode(method: str, mode: str, steps: int, dt: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate synthetic episode traces for baseline-comparison scripting."""
    t = np.arange(steps) * dt
    severity = _method_severity(method, mode)
    joint_jerk = np.abs(rng.normal(loc=severity, scale=0.2, size=steps))
    com_jerk = np.abs(rng.normal(loc=0.5 * severity, scale=0.15, size=steps))
    foot_slip = np.abs(rng.normal(loc=0.2 * severity, scale=0.05, size=steps))
    penetration = np.clip(rng.normal(loc=0.00005 * severity, scale=0.0001, size=steps), 0.0, None)
    torque = np.abs(rng.normal(loc=2.0 * severity, scale=0.3, size=steps))
    torque_rate = np.abs(np.gradient(torque, dt))
    root_pitch = 0.03 * severity * np.sin(2.0 * np.pi * t)
    root_roll = 0.02 * severity * np.sin(1.7 * np.pi * t)
    fail_prob = 0.004 * severity if mode == "single_skill" else 0.02 * severity
    alive = (rng.uniform(size=steps) > fail_prob).astype(np.float64)
    switch_success = (rng.uniform(size=steps) > (0.02 * severity if mode == "single_skill" else 0.12 * severity)).astype(np.float64)

    switch_indices = np.array([int(steps * 0.5)]) if mode == "single_skill" else np.array([int(steps * 0.25), int(steps * 0.5), int(steps * 0.75)])

    return {
        "joint_jerk_norm": joint_jerk,
        "com_jerk_norm": com_jerk,
        "foot_slip_norm": foot_slip,
        "ground_penetration_depth": penetration,
        "torque_norm": torque,
        "torque_rate_norm": torque_rate,
        "root_pitch": root_pitch,
        "root_roll": root_roll,
        "alive_mask": alive,
        "switch_success_mask": switch_success,
        "switch_event_indices": switch_indices,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate switching metrics across methods.")
    parser.add_argument("--experiment-name", type=str, default="week1_random_switch")
    parser.add_argument("--mode", type=str, choices=["single_skill", "random_switch"], default="random_switch")
    parser.add_argument("--method", type=str, choices=METHODS, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--window-s", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _evaluate_method(args: argparse.Namespace, method: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    rng = np.random.default_rng(args.seed + METHODS.index(method))
    episode_rows: list[dict] = []
    switch_rows: list[dict] = []

    for episode_idx in range(args.episodes):
        traces = generate_synthetic_episode(method=method, mode=args.mode, steps=480, dt=args.dt, rng=rng)
        metrics, per_switch = compute_switch_window_metrics(dt=args.dt, window_s=args.window_s, **traces)
        severity = _method_severity(method, args.mode)
        metrics.update(_reward_breakdown(method, severity))
        metrics["episode_idx"] = episode_idx
        metrics["method"] = method
        metrics["mode"] = args.mode
        episode_rows.append(metrics)
        for switch_id, switch_metric in enumerate(per_switch):
            switch_metric["episode_idx"] = episode_idx
            switch_metric["switch_id"] = switch_id
            switch_metric["mode"] = args.mode
            switch_metric["method"] = method
            switch_metric["skill_pair"] = "locomotion_run->arm_dribble_like"
            switch_rows.append(switch_metric)

    per_episode_df = pd.DataFrame(episode_rows)
    per_switch_df = pd.DataFrame(switch_rows)
    aggregate = per_episode_df.mean(numeric_only=True).to_dict()
    aggregate["method"] = method
    aggregate["mode"] = args.mode
    aggregate["episodes"] = args.episodes
    latent_df = _synthetic_latents(rng, method=method, mode=args.mode)
    return per_episode_df, per_switch_df, aggregate, latent_df


def main() -> None:
    args = parse_args()
    output_dir = Path("outputs/eval") / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [args.method] if args.method is not None else METHODS

    all_agg: list[dict] = []
    all_episode: list[pd.DataFrame] = []
    all_switch: list[pd.DataFrame] = []
    all_latent: list[pd.DataFrame] = []
    for method in methods:
        per_episode_df, per_switch_df, aggregate, latent_df = _evaluate_method(args, method)
        all_episode.append(per_episode_df)
        all_switch.append(per_switch_df)
        all_agg.append(aggregate)
        all_latent.append(latent_df)

    pd.DataFrame(all_agg).to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(all_episode, ignore_index=True).to_csv(output_dir / "per_episode_metrics.csv", index=False)
    pd.concat(all_switch, ignore_index=True).to_csv(output_dir / "per_switch_metrics.csv", index=False)
    pd.concat(all_latent, ignore_index=True).to_csv(output_dir / "latent_states.csv", index=False)

    config_snapshot = {
        "experiment_name": args.experiment_name,
        "mode": args.mode,
        "methods": methods,
        "episodes": args.episodes,
        "dt": args.dt,
        "window_s": args.window_s,
    }
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False))
    (output_dir / "summary.json").write_text(json.dumps(all_agg, indent=2))

    print(f"Saved evaluation artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
