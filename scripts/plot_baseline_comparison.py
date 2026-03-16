"""Plot baseline-method comparison figures from evaluation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "fall_rate",
    "mean_survival_time",
    "joint_jerk_integral",
    "com_jerk_integral",
    "foot_slip_integral_during_contact",
    "torque_rate_integral",
    "switch_success_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline comparisons.")
    parser.add_argument("--single-exp", type=str, default="week1_single_skill")
    parser.add_argument("--switch-exp", type=str, default="week1_random_switch")
    parser.add_argument("--output-dir", type=str, default="outputs/plots/baseline_comparison")
    return parser.parse_args()


def _load(exp_name: str) -> pd.DataFrame:
    df = pd.read_csv(Path("outputs/eval") / exp_name / "metrics.csv")
    return df


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    single = _load(args.single_exp)
    switch = _load(args.switch_exp)

    merged = pd.concat([single, switch], ignore_index=True)
    merged.to_csv(out / "baseline_metrics_merged.csv", index=False)

    for metric in METRICS:
        pivot = merged.pivot(index="method", columns="mode", values=metric)
        ax = pivot.plot(kind="bar", figsize=(7, 4), title=f"{metric} by method")
        ax.set_ylabel(metric)
        ax.figure.tight_layout()
        ax.figure.savefig(out / f"{metric}.png", dpi=150)
        plt.close(ax.figure)

    print(f"Saved baseline comparison plots to: {out}")


if __name__ == "__main__":
    main()
