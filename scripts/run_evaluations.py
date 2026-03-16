"""Batch evaluator for all methods and seeds with deterministic naming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

METHODS = ["ppo_cmd", "fullbody_amp", "partwise_raw", "famtp_stage1", "famtp_nobridge", "famtp_full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation sweeps.")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--mode", type=str, default="random_switch", choices=["single_skill", "random_switch"])
    parser.add_argument("--checkpoint-root", type=str, default="logs/rsl_rl")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_dir = Path("outputs/experiments/eval_runs")
    meta_dir.mkdir(parents=True, exist_ok=True)

    records = []
    ckpt_root = Path(args.checkpoint_root)
    for method in args.methods:
        for seed in args.seeds:
            exp_name = f"eval_{args.mode}_method-{method}_seed-{seed}"
            cmd = [
                "python",
                "scripts/eval_switching.py",
                "--experiment-name",
                exp_name,
                "--mode",
                args.mode,
                "--method",
                method,
                "--episodes",
                str(args.episodes),
                "--seed",
                str(seed),
            ]
            # fail clearly when checkpoints are missing (unless dry-run).
            expected_dir = ckpt_root / method
            if not args.dry_run and not expected_dir.exists():
                raise FileNotFoundError(f"Missing checkpoint directory for method {method}: {expected_dir}")
            records.append({"exp_name": exp_name, "method": method, "seed": seed, "cmd": cmd})
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)

    (meta_dir / "eval_runs.json").write_text(json.dumps(records, indent=2))
    print(f"Saved metadata to {meta_dir / 'eval_runs.json'}")


if __name__ == "__main__":
    main()
