#!/usr/bin/env python3
"""Build a concise Markdown report from an SVDD sensitivity analysis CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def finite(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-by-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=48)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.summary_by_run.open(encoding="utf-8")))
    for row in rows:
        row["_tacc"] = finite(row, "final_tacc_mean")
        row["_tpr"] = finite(row, "final_tpr_mean")
        row["_fpr"] = finite(row, "final_fpr_mean")
        row["_dpr"] = finite(row, "final_dpr_mean")
        row["_asr"] = finite(row, "final_backdoor_asr_mean")

    by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)

    def line(row: dict[str, str]) -> str:
        return (
            f"| {row['phase1_rounds']} | {float(row['malicious_ratio'])*100:.0f}% | "
            f"{row['mode']} | {pct(row['_tacc'])} | {pct(row['_tpr'])} | "
            f"{pct(row['_fpr'])} | {pct(row['_dpr'])} |"
        )

    ranked_tacc = sorted(rows, key=lambda row: row["_tacc"] or -1.0, reverse=True)
    ranked_fpr = sorted(
        [row for row in rows if row["_tpr"] is not None and row["_tpr"] >= 0.99],
        key=lambda row: (row["_fpr"] if row["_fpr"] is not None else 1.0, -(row["_tacc"] or 0.0)),
    )
    output: list[str] = [
        "# AGNews AE-SVDD 敏感性矩阵实验报告",
        "",
        f"- 完整运行数：{len(rows)}/{args.expected_runs}",
        "- 数据集：AGNews；客户端数：100；数据划分：IID；总轮数：300",
        "- 客户端优化：`client_lr=0.1`，`client_weight_decay=0.0005`，本地 epoch=1，batch size=64",
        "- 攻击：`mix`，混合成员为 `lf,bd,gn`；随机种子：42",
        "- 评分模式：`recon`（仅重建）、`combined`（重建+SVDD）、`svdd`（仅SVDD）",
        "- 每个结果为最后 10 轮均值；TPR/FPR 是客户端检测指标。",
        "",
        "## 按评分模式汇总",
        "",
        "| 模式 | runs | TACC | TPR | FPR | DPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("recon", "combined", "svdd"):
        members = by_mode.get(mode, [])
        output.append(
            f"| {mode} | {len(members)} | {pct(mean([r['_tacc'] for r in members]))} | "
            f"{pct(mean([r['_tpr'] for r in members]))} | {pct(mean([r['_fpr'] for r in members]))} | "
            f"{pct(mean([r['_dpr'] for r in members]))} |"
        )
    output += [
        "",
        "## 最高干净准确率配置",
        "",
        "| P1轮数 | 恶意比例 | 模式 | TACC | TPR | FPR | DPR |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    output.extend(line(row) for row in ranked_tacc[:10])
    output += [
        "",
        "## TPR≥99% 时的最低 FPR 配置",
        "",
        "| P1轮数 | 恶意比例 | 模式 | TACC | TPR | FPR | DPR |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    output.extend(line(row) for row in ranked_fpr[:10])
    output += [
        "",
        "## 解释与限制",
        "",
        "- 当前矩阵固定 Mix 攻击和 seed=42，因此结果是单 seed、单攻击协议下的敏感性分析，不是跨攻击泛化结论。",
        "- AGNews 的文本 Mix 攻击没有独立触发器测试集，`backdoor_asr` 不适用，报告中不将其当作 ASR 结论。",
        "- 结果文件由 `tools/analyze_svdd_sensitivity_matrix.py` 从原始逐轮 JSON 生成；完整矩阵应为 4×4×3=48 个配置。",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
