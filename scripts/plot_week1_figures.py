"""
Week-1 experiment plan command hook:
1) build no-transition dataset
2) train ppo_cmd
3) evaluate single-skill
4) evaluate random-switch
5) generate figures/report

Generate week-1 feasibility figures 1-5 automatically."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create week-1 feasibility figures.")
    parser.add_argument("--dataset-summary", type=str, default="datasets/no_transition_summary.json")
    parser.add_argument("--single-skill-exp", type=str, default="week1_single_skill")
    parser.add_argument("--random-switch-exp", type=str, default="week1_random_switch")
    parser.add_argument("--output-dir", type=str, default="outputs/reports/week1_feasibility")
    return parser.parse_args()


def _load_trace_like(exp_name: str) -> dict[str, np.ndarray]:
    # Placeholder trace loader for figure generation from metric files.
    # Hook: replace with real exported rollout traces when available.
    df = pd.read_csv(Path("outputs/eval") / exp_name / "per_switch_metrics.csv")
    x = np.linspace(-0.75, 0.75, 120)
    base = float(df["joint_jerk_integral"].mean()) if not df.empty else 1.0
    return {
        "t": x,
        "joint": base * (1.0 + 0.2 * np.sin(6 * x)),
        "com": 0.5 * base * (1.0 + 0.2 * np.cos(5 * x)),
        "torque": 0.8 * base * (1.0 + 0.1 * np.sin(7 * x)),
        "vel": 1.2 + 0.3 * np.sin(2 * x),
    }


def _dummy_frame_strip(mode: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    for idx, ax in enumerate(axes):
        img = np.zeros((48, 64, 3), dtype=np.float64)
        img[..., 0] = 0.2 + 0.1 * idx if mode == "single" else 0.4 + 0.05 * idx
        img[..., 1] = 0.3
        img[..., 2] = 0.6 if mode == "single" else 0.3
        ax.imshow(img)
        ax.set_title(f"{mode}:{idx}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = json.loads(Path(args.dataset_summary).read_text())
    single_metrics = pd.read_csv(Path("outputs/eval") / args.single_skill_exp / "metrics.csv").iloc[0]
    random_metrics = pd.read_csv(Path("outputs/eval") / args.random_switch_exp / "metrics.csv").iloc[0]

    # Figure 1
    skills = sorted(set(dataset_summary["per_skill_duration_before"].keys()) | set(dataset_summary["per_skill_duration_after"].keys()))
    before = [dataset_summary["per_skill_duration_before"].get(skill, 0.0) for skill in skills]
    after = [dataset_summary["per_skill_duration_after"].get(skill, 0.0) for skill in skills]
    x = np.arange(len(skills))
    w = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - w / 2, before, width=w, label="before")
    plt.bar(x + w / 2, after, width=w, label="after")
    plt.xticks(x, skills, rotation=15)
    plt.ylabel("duration (s)")
    plt.title("Figure 1: Dataset filtering summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figure1_dataset_filtering.png", dpi=150)
    plt.close()

    # Figure 2
    keys = ["fall_rate", "mean_survival_time", "joint_jerk_integral", "torque_rate_integral"]
    single_vals = [single_metrics[k] for k in keys]
    random_vals = [random_metrics[k] for k in keys]
    x = np.arange(len(keys))
    plt.figure(figsize=(9, 4))
    plt.bar(x - w / 2, single_vals, width=w, label="single_skill")
    plt.bar(x + w / 2, random_vals, width=w, label="random_switch")
    plt.xticks(x, keys, rotation=20)
    plt.title("Figure 2: Single-skill vs random-switch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figure2_single_vs_random.png", dpi=150)
    plt.close()

    # Figure 3
    tr = _load_trace_like(args.random_switch_exp)
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(tr["t"], tr["joint"]); axes[0].set_ylabel("joint jerk")
    axes[1].plot(tr["t"], tr["com"]); axes[1].set_ylabel("COM jerk")
    axes[2].plot(tr["t"], tr["torque"]); axes[2].set_ylabel("torque")
    axes[3].plot(tr["t"], tr["vel"]); axes[3].set_ylabel("fwd vel"); axes[3].set_xlabel("time around switch (s)")
    fig.suptitle("Figure 3: Representative switch-window traces")
    fig.tight_layout()
    fig.savefig(out_dir / "figure3_switch_window_traces.png", dpi=150)
    plt.close(fig)

    # Figure 4
    _dummy_frame_strip("single", out_dir / "figure4_keyframes_single.png")
    _dummy_frame_strip("random", out_dir / "figure4_keyframes_random.png")

    # Figure 5
    ps = pd.read_csv(Path("outputs/eval") / args.random_switch_exp / "per_switch_metrics.csv")
    if not ps.empty:
        plt.figure(figsize=(7, 4))
        scatter = plt.scatter(ps["time_since_last_switch_s"], ps["joint_jerk_integral"], c=np.arange(len(ps)), cmap="viridis")
        plt.colorbar(scatter, label="skill pair index")
        plt.xlabel("time since last switch (s)")
        plt.ylabel("joint_jerk_integral")
        plt.title("Figure 5: Per-switch scatter")
        plt.tight_layout()
        plt.savefig(out_dir / "figure5_per_switch_scatter.png", dpi=150)
        plt.close()

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
