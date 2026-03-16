"""Plot FaMTP stage-1 latent analyses (phase histograms + PCA projection)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FaMTP stage-1 latent analyses.")
    parser.add_argument("--experiment-name", type=str, default="week1_random_switch")
    parser.add_argument("--output-dir", type=str, default="outputs/plots/famtp_stage1_latents")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    latent_path = Path("outputs/eval") / args.experiment_name / "latent_states.csv"
    df = pd.read_csv(latent_path)
    df = df[df["method"] == "famtp_stage1"]

    # Figure 1: phase histogram proxy using atan2(z2, z1).
    df["phase"] = np.arctan2(df["z2"], df["z1"])
    plt.figure(figsize=(8, 4))
    for skill, group in df.groupby("skill_label"):
        plt.hist(group["phase"], bins=30, alpha=0.5, label=skill)
    plt.title("FaMTP Stage-1 latent phase histogram by skill")
    plt.xlabel("phase (rad)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "latent_phase_histograms.png", dpi=150)
    plt.close()

    # Figure 2: PCA projection (2D) from z1,z2 (already 2D proxy).
    plt.figure(figsize=(6, 5))
    for skill, group in df.groupby("skill_label"):
        plt.scatter(group["z1"], group["z2"], s=10, alpha=0.4, label=skill)
    plt.title("FaMTP Stage-1 latent projection (PCA proxy)")
    plt.xlabel("latent axis 1")
    plt.ylabel("latent axis 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "latent_projection_pca_proxy.png", dpi=150)
    plt.close()

    print(f"Saved latent plots to: {out}")


if __name__ == "__main__":
    main()
