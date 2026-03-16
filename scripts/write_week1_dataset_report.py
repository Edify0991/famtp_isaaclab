"""Write week1 dataset markdown summary report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write week1 dataset summary markdown.")
    parser.add_argument("--report-root", type=str, default="outputs/reports/week1_feasibility")
    parser.add_argument("--scan-summary", type=str, default="datasets/reports/g1_amass_schema_summary.json")
    parser.add_argument("--candidate-index", type=str, default="datasets/canonical/week1_candidate_motion_index.json")
    parser.add_argument("--motion-index", type=str, default="datasets/canonical/week1_motion_index.json")
    parser.add_argument("--no-transition-index", type=str, default="datasets/canonical/week1_no_transition_motion_index.json")
    parser.add_argument("--filter-summary", type=str, default="datasets/reports/week1_filtering_summary.json")
    return parser.parse_args()


def _read_json(path: str) -> dict | list:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _duration(rows: list[dict]) -> float:
    return float(sum(float(r.get("duration", 0.0)) for r in rows))


def main() -> None:
    args = parse_args()
    root = Path(args.report_root)
    root.mkdir(parents=True, exist_ok=True)

    scan = _read_json(args.scan_summary)
    cand = _read_json(args.candidate_index)
    week1 = _read_json(args.motion_index)
    nt = _read_json(args.no_transition_index)
    filt = _read_json(args.filter_summary)

    cand = cand if isinstance(cand, list) else []
    week1 = week1 if isinstance(week1, list) else []
    nt = nt if isinstance(nt, list) else []

    lines = [
        "# Week1 Dataset Summary",
        "",
        "## 1. 数据来源",
        "- 本地 Hugging Face 数据集：`ember-lab-berkeley/AMASS_Retargeted_for_G1`",
        "- BABEL 标注用于技能候选挖掘（当前阶段聚焦先看后筛）。",
        "",
        "## 2. G1 数据 schema 扫描结果",
        f"- key 频次 Top: {list((scan.get('key_frequency') or {}).items())[:10]}",
        f"- key 组合 Top: {scan.get('keyset_combinations', [])[:5]}",
        f"- fps 统计: {scan.get('fps_counts', {})}",
        "",
        "## 3. viewer 回放支持情况",
        "- 支持自动字段推断与 canonical clip 映射。",
        "- 若字段不足，使用 inspection fallback（时间轴/维度/轨迹导出）。",
        "- 支持截图、关键帧条带图、root trajectory 导出。",
        "",
        "## 4. 三个 skill 池定义",
        "- locomotion_runlike",
        "- dribble_like",
        "- stop_shoot_like",
        "",
        "## 5. 候选匹配数量",
        f"- candidate clips: {len(cand)}",
        "",
        "## 6. 筛选后 clip 数量和总时长",
        f"- week1 clips: {len(week1)}",
        f"- week1 total duration: {_duration(week1):.2f}s",
        "",
        "## 7. no-transition 过滤规则",
        f"- filter summary: {filt}",
        f"- no-transition clips: {len(nt)}",
        f"- no-transition total duration: {_duration(nt):.2f}s",
        "",
        "## 8. 当前局限和未完成项",
        "- 仍需按实际 G1 npz 字段完善精确关节映射。",
        "- BABEL 与本地文件匹配目前以文件名/序列名启发式为主。",
        "",
        "## 9. 下一步",
        "- 在 week1_no_transition_motion_index.json 基础上接 ppo_cmd。",
        "- 做 single_skill 与 random_switch 评测。",
    ]

    out = root / "week1_dataset_summary.md"
    out.write_text("\n".join(lines))
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
