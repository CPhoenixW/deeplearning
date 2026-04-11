from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _avg(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def summarize_log(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["meta"]
    rounds = data["round_metrics"]
    last10 = rounds[-10:] if len(rounds) >= 10 else rounds
    return {
        "task": meta.get("task_name"),
        "attack": meta.get("attack"),
        "defense": meta.get("defense"),
        "rounds": int(meta.get("total_rounds", len(rounds))),
        "avg_test_acc_last10": _avg([float(r["test_acc"]) for r in last10]),
        "avg_tpr_last10": _avg([float(r["malicious_detection_rate"]) for r in last10]),
        "avg_fpr_last10": _avg([float(r["benign_false_positive_rate"]) for r in last10]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", required=True)
    p.add_argument("--out-md", default="REPORT_LAST10.md")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    files = sorted(log_dir.glob("*.json"))
    rows = [summarize_log(f) for f in files]
    rows.sort(key=lambda x: (str(x["task"]), str(x["attack"]), str(x["defense"])))

    md: List[str] = []
    md.append("## Final 10-round Average Metrics")
    md.append("")
    md.append(f"- Log dir: `{log_dir}`")
    md.append(f"- Total runs summarized: **{len(rows)}**")
    md.append("")
    md.append("| Task | Attack | Defense | Rounds | Avg Test Acc (last10) | Avg TPR (last10) | Avg FPR (last10) |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['task']} | {r['attack']} | {r['defense']} | {r['rounds']} | "
            f"{r['avg_test_acc_last10']:.4f} | {r['avg_tpr_last10']:.4f} | {r['avg_fpr_last10']:.4f} |"
        )
    out_path = Path(args.out_md)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
