#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return mean(values)


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def parse_one_log(file_path: Path, last_n: int) -> Optional[Dict[str, object]]:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] 读取失败，跳过 {file_path.name}: {exc}")
        return None

    meta = data.get("meta", {})
    round_metrics = data.get("round_metrics", [])
    if not round_metrics:
        print(f"[WARN] round_metrics 为空，跳过 {file_path.name}")
        return None

    attack = meta.get("attack")
    defense = meta.get("defense")
    task_name = meta.get("task_name")
    if not attack or not defense or not task_name:
        print(f"[WARN] 元信息缺失，跳过 {file_path.name}")
        return None

    tail = round_metrics[-last_n:]
    acc_values = [x.get("test_acc") for x in tail if isinstance(x.get("test_acc"), (int, float))]
    dar_values = [x.get("dar") for x in tail if isinstance(x.get("dar"), (int, float))]
    dpr_values = [x.get("dpr") for x in tail if isinstance(x.get("dpr"), (int, float))]

    return {
        "task_name": task_name,
        "attack": attack,
        "defense": defense,
        "round_count_used": len(tail),
        "avg_acc": safe_mean(acc_values),
        "avg_dar": safe_mean(dar_values),
        "avg_dpr": safe_mean(dpr_values),
        "source_file": file_path.name,
    }


def build_table(rows: List[Dict[str, object]], task_filter: str) -> str:
    filtered = [r for r in rows if r["task_name"] == task_filter]
    if not filtered:
        return f"未找到 task_name={task_filter} 的日志。"

    by_attack: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in filtered:
        by_attack[row["attack"]].append(row)

    lines: List[str] = []
    for attack in sorted(by_attack.keys()):
        lines.append(f"\n=== 攻击: {attack} ===")
        lines.append(f"{'Defense':<10} {'Avg_ACC':>10} {'Avg_DAR':>10} {'Avg_DPR':>10} {'RoundsUsed':>10}")
        lines.append("-" * 56)
        attack_rows = sorted(by_attack[attack], key=lambda x: str(x["defense"]))
        for r in attack_rows:
            lines.append(
                f"{str(r['defense']):<10} "
                f"{fmt(r['avg_acc']):>10} "
                f"{fmt(r['avg_dar']):>10} "
                f"{fmt(r['avg_dpr']):>10} "
                f"{str(r['round_count_used']):>10}"
            )
    return "\n".join(lines)


def export_csv(rows: List[Dict[str, object]], out_path: Path, task_filter: str) -> None:
    import csv

    filtered = [r for r in rows if r["task_name"] == task_filter]
    headers = [
        "task_name",
        "attack",
        "defense",
        "avg_acc_last_n",
        "avg_dar_last_n",
        "avg_dpr_last_n",
        "round_count_used",
        "source_file",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in sorted(filtered, key=lambda x: (str(x["attack"]), str(x["defense"]))):
            writer.writerow(
                {
                    "task_name": r["task_name"],
                    "attack": r["attack"],
                    "defense": r["defense"],
                    "avg_acc_last_n": r["avg_acc"],
                    "avg_dar_last_n": r["avg_dar"],
                    "avg_dpr_last_n": r["avg_dpr"],
                    "round_count_used": r["round_count_used"],
                    "source_file": r["source_file"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析 log 文件夹中 AG News 各防御方法在不同攻击下最后 N 轮平均 ACC/DAR/DPR。"
    )
    parser.add_argument("--log-dir", default="log", help="日志目录，默认: log")
    parser.add_argument("--task", default="ag_news", help="任务名过滤，默认: ag_news")
    parser.add_argument("--last-n", type=int, default=10, help="取最后 N 轮平均，默认: 10")
    parser.add_argument("--csv-out", default="", help="可选，导出 CSV 路径")
    args = parser.parse_args()

    if args.last_n <= 0:
        raise ValueError("--last-n 必须大于 0")

    log_dir = Path(args.log_dir)
    if not log_dir.exists() or not log_dir.is_dir():
        raise FileNotFoundError(f"日志目录不存在: {log_dir}")

    pattern = f"{args.task}__*.json"
    files = sorted(log_dir.glob(pattern))
    if not files:
        print(f"在 {log_dir} 中未找到匹配 {pattern} 的日志文件。")
        return

    rows: List[Dict[str, object]] = []
    for file_path in files:
        result = parse_one_log(file_path, args.last_n)
        if result is not None:
            rows.append(result)

    if not rows:
        print("没有可用日志可分析。")
        return

    print(build_table(rows, args.task))

    if args.csv_out:
        out_path = Path(args.csv_out)
        export_csv(rows, out_path, args.task)
        print(f"\nCSV 已导出到: {out_path}")


if __name__ == "__main__":
    main()
