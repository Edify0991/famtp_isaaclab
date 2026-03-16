"""Mine week-1 skill candidates by matching BABEL labels with local G1 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SKILL_RULES = {
    "locomotion_runlike": [
        "walk",
        "run",
        "jog",
        "sprint",
        "walk forward",
        "walk back and forth",
        "forward locomotion",
    ],
    "dribble_like": [
        "dribble",
        "dribble basketball",
        "dribble basketball with left hand",
        "dribble basketball with right hand",
        "bounce ball",
        "bounce basketball",
        "play basketball bounce ball",
    ],
    "stop_shoot_like": [
        "stop",
        "run stop",
        "slow to a stop",
        "come to a stop",
        "stop and stand",
        "aim",
        "aim with right hand",
        "shoot basketball",
        "shoot a basketball",
        "throw a basketball",
        "throw ball",
        "baseball pitch",
        "reach up",
        "stretch arms overhead",
        "extend both arms up",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine week1 skills from BABEL metadata.")
    parser.add_argument("--babel-json", type=str, required=True)
    parser.add_argument("--g1-root", type=str, required=True)
    parser.add_argument("--output-index", type=str, default="datasets/canonical/week1_candidate_motion_index.json")
    parser.add_argument("--output-summary-json", type=str, default="datasets/reports/week1_skill_mining_summary.json")
    parser.add_argument("--output-summary-md", type=str, default="datasets/reports/week1_skill_mining_summary.md")
    return parser.parse_args()


def _flatten_labels(entry: dict) -> list[str]:
    texts = []
    for key in ["babel_raw_label", "babel_proc_label", "raw_label", "proc_label", "label", "act_cat"]:
        value = entry.get(key)
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            texts.extend([str(v) for v in value])
    # BABEL nested styles
    for key in ["frame_ann", "seq_ann"]:
        ann = entry.get(key)
        if isinstance(ann, dict):
            labels = ann.get("labels", [])
            for obj in labels:
                if isinstance(obj, dict):
                    for k in ["raw_label", "proc_label", "act_cat"]:
                        if k in obj and obj[k]:
                            texts.append(str(obj[k]))
    return [t.lower() for t in texts]


def _match_skill(labels: list[str]) -> tuple[str | None, str | None, float]:
    best_skill = None
    best_label = None
    best_score = 0.0
    for skill, patterns in SKILL_RULES.items():
        for p in patterns:
            for l in labels:
                if p in l:
                    score = min(1.0, max(len(p) / max(len(l), 1), 0.35))
                    if score > best_score:
                        best_score = score
                        best_skill = skill
                        best_label = l
    return best_skill, best_label, best_score


def _g1_file_map(root: Path) -> dict[str, str]:
    mapping = {}
    for p in root.rglob("*.npz"):
        mapping[p.stem.lower()] = p.as_posix()
    return mapping


def _find_match_path(entry: dict, g1_map: dict[str, str]) -> str | None:
    candidates = []
    for key in ["feat_p", "url", "seq_name", "amass_seq", "clip_id", "id"]:
        if key in entry and entry[key]:
            candidates.append(str(entry[key]))
    joined = " ".join(candidates).lower()
    for stem, path in g1_map.items():
        if stem in joined or joined in stem:
            return path
    return None


def main() -> None:
    args = parse_args()
    babel = json.loads(Path(args.babel_json).read_text())
    entries = list(babel.values()) if isinstance(babel, dict) else babel
    g1_map = _g1_file_map(Path(args.g1_root))

    out_rows = []
    for idx, entry in enumerate(entries):
        labels = _flatten_labels(entry)
        skill, matched_label, conf = _match_skill(labels)
        if not skill:
            continue
        matched = _find_match_path(entry, g1_map)
        out_rows.append(
            {
                "clip_id": str(entry.get("clip_id", entry.get("id", f"babel_{idx}"))),
                "source_file": str(entry.get("feat_p", entry.get("url", "unknown"))),
                "matched_g1_file_path": matched,
                "skill_label": skill,
                "babel_raw_label": entry.get("raw_label") or matched_label,
                "babel_proc_label": entry.get("proc_label") or matched_label,
                "act_cat": entry.get("act_cat", ""),
                "start_time": float(entry.get("start_t", entry.get("start_time", 0.0))),
                "end_time": float(entry.get("end_t", entry.get("end_time", 0.0))),
                "duration": float(entry.get("end_t", entry.get("end_time", 0.0))) - float(entry.get("start_t", entry.get("start_time", 0.0))),
                "match_confidence": float(conf if matched else conf * 0.5),
                "notes": "matched by filename stem" if matched else "no g1 filename match",
            }
        )

    out_path = Path(args.output_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, indent=2))

    by_skill = {}
    for row in out_rows:
        by_skill[row["skill_label"]] = by_skill.get(row["skill_label"], 0) + 1

    summary = {
        "num_candidates": len(out_rows),
        "by_skill": by_skill,
        "matched_g1_count": sum(1 for r in out_rows if r.get("matched_g1_file_path")),
    }
    summary_json = Path(args.output_summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2))

    md = ["# Week1 skill mining summary", "", f"- candidates: {summary['num_candidates']}", f"- matched_g1_count: {summary['matched_g1_count']}", "", "## by skill"]
    for k, v in by_skill.items():
        md.append(f"- {k}: {v}")
    Path(args.output_summary_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary_md).write_text("\n".join(md))

    print(f"Saved candidate index: {out_path}")


if __name__ == "__main__":
    main()
