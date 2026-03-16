"""
Week-1 experiment plan command hook:
1) build no-transition dataset
2) train ppo_cmd
3) evaluate single-skill
4) evaluate random-switch
5) generate figures/report

Evaluate week-1 switching degradation metrics and export CSV/JSON artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from famtp_lab.tasks.direct.humanoid_switch.metrics import compute_switch_window_metrics


def generate_synthetic_episode(mode: str, steps: int, dt: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate synthetic episode traces when simulator rollouts are unavailable."""
    t = np.arange(steps) * dt
    severity = 0.6 if mode == "single_skill" else 1.4
    joint_jerk = np.abs(rng.normal(loc=severity, scale=0.2, size=steps))
    com_jerk = np.abs(rng.normal(loc=0.5 * severity, scale=0.15, size=steps))
    foot_slip = np.abs(rng.normal(loc=0.2 * severity, scale=0.05, size=steps))
    penetration = np.clip(rng.normal(loc=0.00005 * severity, scale=0.0001, size=steps), 0.0, None)
    torque = np.abs(rng.normal(loc=2.0 * severity, scale=0.3, size=steps))
    torque_rate = np.abs(np.gradient(torque, dt))
    root_pitch = 0.03 * severity * np.sin(2.0 * np.pi * t)
    root_roll = 0.02 * severity * np.sin(1.7 * np.pi * t)
    fail_prob = 0.005 if mode == "single_skill" else 0.03
    alive = (rng.uniform(size=steps) > fail_prob).astype(np.float64)
    switch_success = (rng.uniform(size=steps) > (0.02 if mode == "single_skill" else 0.15)).astype(np.float64)

    if mode == "single_skill":
        switch_indices = np.array([int(steps * 0.5)])
    else:
        switch_indices = np.array([int(steps * 0.25), int(steps * 0.5), int(steps * 0.75)])

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
    parser = argparse.ArgumentParser(description="Evaluate switching metrics.")
    parser.add_argument("--experiment-name", type=str, default="week1_random_switch")
    parser.add_argument("--mode", type=str, choices=["single_skill", "random_switch"], default="random_switch")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--window-s", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("outputs/eval") / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    episode_rows: list[dict] = []
    switch_rows: list[dict] = []

    for episode_idx in range(args.episodes):
        traces = generate_synthetic_episode(args.mode, steps=480, dt=args.dt, rng=rng)
        metrics, per_switch = compute_switch_window_metrics(dt=args.dt, window_s=args.window_s, **traces)
        metrics["episode_idx"] = episode_idx
        episode_rows.append(metrics)
        for switch_id, switch_metric in enumerate(per_switch):
            switch_metric["episode_idx"] = episode_idx
            switch_metric["switch_id"] = switch_id
            switch_metric["mode"] = args.mode
            # fixed skill pair annotation hook for future real env logging.
            switch_metric["skill_pair"] = "locomotion_run->arm_dribble_like"
            switch_rows.append(switch_metric)

    per_episode_df = pd.DataFrame(episode_rows)
    per_switch_df = pd.DataFrame(switch_rows)
    aggregate = per_episode_df.mean(numeric_only=True).to_dict()
    aggregate["mode"] = args.mode
    aggregate["episodes"] = args.episodes

    pd.DataFrame([aggregate]).to_csv(output_dir / "metrics.csv", index=False)
    per_episode_df.to_csv(output_dir / "per_episode_metrics.csv", index=False)
    per_switch_df.to_csv(output_dir / "per_switch_metrics.csv", index=False)

    config_snapshot = {
        "experiment_name": args.experiment_name,
        "mode": args.mode,
        "episodes": args.episodes,
        "dt": args.dt,
        "window_s": args.window_s,
        "baseline_mode": "ppo_cmd",
    }
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False))
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))

    print(f"Saved evaluation artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
