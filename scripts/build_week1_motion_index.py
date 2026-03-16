"""Build filtered week1 motion index from mined candidate entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_RANGE = (20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build week1 motion index.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="datasets/canonical/week1_motion_index.json")
    parser.add_argument("--min-duration", type=float, default=0.8)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(Path(args.input).read_text())

    filtered = []
    for r in rows:
        if float(r.get("duration", 0.0)) < args.min_duration:
            continue
        path = r.get("matched_g1_file_path")
        if not path or not Path(path).exists():
            continue
        if float(r.get("match_confidence", 0.0)) < args.min_confidence:
            continue
        filtered.append(r)

    by_skill: dict[str, list[dict]] = {}
    for r in filtered:
        by_skill.setdefault(r["skill_label"], []).append(r)

    selected: list[dict] = []
    for skill, arr in by_skill.items():
        arr.sort(key=lambda x: (-float(x.get("match_confidence", 0.0)), -float(x.get("duration", 0.0))))
        keep = arr[: TARGET_RANGE[1]]
        if len(keep) < TARGET_RANGE[0]:
            keep = arr
        selected.extend(keep)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2))
    print(f"Saved week1 index: {out} (clips={len(selected)})")


if __name__ == "__main__":
    main()
