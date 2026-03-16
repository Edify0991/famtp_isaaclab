"""Plot reward decomposition for baseline methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward decomposition across methods.")
    parser.add_argument("--experiment-name", type=str, default="week1_random_switch")
    parser.add_argument("--output-dir", type=str, default="outputs/plots/famtp_stage1_latents")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(Path("outputs/eval") / args.experiment_name / "metrics.csv")
    cols = ["reward_task", "reward_latent_part", "reward_coupling"]
    plot_df = metrics[["method", *cols]].set_index("method")

    ax = plot_df.plot(kind="bar", stacked=True, figsize=(8, 4), title="Reward decomposition by method")
    ax.set_ylabel("reward")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "reward_decomposition.png", dpi=150)
    plt.close(ax.figure)

    print(f"Saved reward decomposition plot to: {out}")


if __name__ == "__main__":
    main()
