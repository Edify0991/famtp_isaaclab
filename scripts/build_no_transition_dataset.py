"""
Week-1 experiment plan command hook:
1) build no-transition dataset
2) train ppo_cmd
3) evaluate single-skill
4) evaluate random-switch
5) generate figures/report

Build week-1 no-transition motion dataset index."""

from __future__ import annotations

import argparse

from famtp_lab.motion.motion_loader import DEFAULT_SKILLS, build_no_transition_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-transition motion index.")
    parser.add_argument("--input", type=str, default="datasets/motion_index.json")
    parser.add_argument("--output", type=str, default="datasets/no_transition_motion_index.json")
    parser.add_argument("--summary", type=str, default="datasets/no_transition_summary.json")
    parser.add_argument("--boundary-window-s", type=float, default=0.25)
    parser.add_argument(
        "--skills",
        nargs="*",
        default=DEFAULT_SKILLS,
        help="Skills to preserve in no-transition dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_no_transition_dataset(
        input_index_path=args.input,
        output_index_path=args.output,
        summary_path=args.summary,
        boundary_window_s=args.boundary_window_s,
        allowed_skills=args.skills,
    )
    print("No-transition dataset built.")
    print(f"clips: {summary['total_clips_before']} -> {summary['total_clips_after']}")


if __name__ == "__main__":
    main()
