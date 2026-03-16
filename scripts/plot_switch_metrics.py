"""Plot main switch metrics comparison bars."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRICS = ["fall_rate", "mean_survival_time", "joint_jerk_integral", "com_jerk_integral", "switch_success_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-exp", type=str, default="week1_single_skill")
    parser.add_argument("--switch-exp", type=str, default="week1_random_switch")
    parser.add_argument("--output-dir", type=str, default="outputs/plots/baseline_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    single = pd.read_csv(Path("outputs/eval") / args.single_exp / "metrics.csv")
    switch = pd.read_csv(Path("outputs/eval") / args.switch_exp / "metrics.csv")
    merged = pd.concat([single, switch], ignore_index=True)

    for m in METRICS:
        piv = merged.pivot(index="method", columns="mode", values=m)
        ax = piv.plot(kind="bar", figsize=(7, 4), title=f"{m} comparison")
        ax.set_ylabel(m)
        ax.figure.tight_layout()
        ax.figure.savefig(out / f"main_{m}.png", dpi=150)
        plt.close(ax.figure)


if __name__ == "__main__":
    main()
