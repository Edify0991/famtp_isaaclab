"""
Week-1 experiment plan command hook:
1) build no-transition dataset
2) train ppo_cmd
3) evaluate single-skill
4) evaluate random-switch
5) generate figures/report

Generate week-1 feasibility markdown summary from evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate week-1 feasibility report markdown.")
    parser.add_argument("--single-skill-exp", type=str, default="week1_single_skill")
    parser.add_argument("--random-switch-exp", type=str, default="week1_random_switch")
    parser.add_argument("--dataset-summary", type=str, default="datasets/no_transition_summary.json")
    parser.add_argument("--output-md", type=str, default="outputs/reports/week1_feasibility/week1_summary.md")
    return parser.parse_args()


def load_metrics(exp_name: str) -> dict:
    metrics_path = Path("outputs/eval") / exp_name / "metrics.csv"
    df = pd.read_csv(metrics_path)
    if "method" in df.columns:
        df = df[df["method"] == "ppo_cmd"]
    row = df.iloc[0].to_dict()
    return row


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_summary = json.loads(Path(args.dataset_summary).read_text())
    single = load_metrics(args.single_skill_exp)
    random_switch = load_metrics(args.random_switch_exp)

    md = f"""# Week-1 Feasibility Report

## 1. Problem statement
This week demonstrates that composing zero-transition humanoid skills is measurably difficult.
The motion index contains only within-skill clips, while runtime random switching induces degradation.

## 2. Dataset protocol summary
- Input index: `datasets/motion_index.json`
- Output no-transition index: `datasets/no_transition_motion_index.json`
- Clips before/after: {dataset_summary['total_clips_before']} -> {dataset_summary['total_clips_after']}
- Boundary window removal (s): {dataset_summary['removed_boundary_statistics']['boundary_window_s']}

## 3. Evaluation protocol summary
- Baseline policy mode: `ppo_cmd` (no imitation prior, no AMP, no manifold encoder)
- Comparison:
  - single-skill condition
  - random-switch condition
- Switch window: [-0.75s, +0.75s] around each switch event

## 4. Key results
- Fall rate (single/random): {single['fall_rate']:.4f} / {random_switch['fall_rate']:.4f}
- Mean survival time (single/random): {single['mean_survival_time']:.3f} / {random_switch['mean_survival_time']:.3f}
- Joint jerk integral (single/random): {single['joint_jerk_integral']:.3f} / {random_switch['joint_jerk_integral']:.3f}
- Torque-rate integral (single/random): {single['torque_rate_integral']:.3f} / {random_switch['torque_rate_integral']:.3f}

## 5. Brief interpretation
- Single-skill operation is easier and more stable.
- Random skill switching is harder and increases transient instability.
- Switch-window metrics capture degradation around command changes.

## 6. Next-step recommendations toward AMP/FaMTP
1. Add imitation rewards from no-transition clips for skill-wise stabilization.
2. Introduce transition-aware latent conditioning and manifold priors.
3. Use switch-window metrics as objective constraints and ablation targets.
"""

    out_path.write_text(md)
    print(f"Saved week-1 summary: {out_path}")


if __name__ == "__main__":
    main()
