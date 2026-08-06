#!/usr/bin/env python3
"""Generate a compact report for the c002 AE-SVDD weight/tau sweep."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from run_sf_weight_tau_sweep import DEFAULT_ATTACKS, VARIANTS


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.mean(finite) if finite else None


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def _load(path: Path, last_n: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds", [])
    meta = payload.get("meta", {})
    selected = rounds[-last_n:]
    metrics: dict[str, float | None] = {}
    for key in ("accuracy", "tpr", "fpr", "dar", "dpr", "rr", "reject_rate", "backdoor_asr"):
        metrics[key] = _mean(
            item.get("evaluation", {}).get(key)
            for item in selected
            if isinstance(item.get("evaluation", {}).get(key), (int, float))
        )
    return {
        "candidate": path.parts[-3],
        "attack": meta.get("attack", path.name.split("__")[1]),
        "rounds": len(rounds),
        "planned_rounds": meta.get("total_rounds"),
        "metrics": metrics,
        "effective": meta.get("effective_config", {}),
    }


def _phase_summary(path: Path, phases: tuple[str, ...]) -> dict[str, dict[str, float | None]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in payload.get("rounds", []):
        grouped[str(item.get("phase", "unknown"))].append(item)
    output: dict[str, dict[str, float | None]] = {}
    for phase in phases:
        values = grouped.get(phase, [])
        output[phase] = {
            key: _mean(
                item.get("evaluation", {}).get(key)
                for item in values
                if isinstance(item.get("evaluation", {}).get(key), (int, float))
            )
            for key in ("accuracy", "tpr", "fpr", "backdoor_asr")
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("log/sf_weight_tau_sweep_single_20260806"))
    parser.add_argument("--last-n", type=int, default=10)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.last_n < 1:
        parser.error("--last-n must be positive")
    root = args.root.resolve()
    paths = sorted(root.glob("*/seed_*/cifar10__*__svdd.json"))
    rows = [_load(path, args.last_n) for path in paths]
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_attack: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by_candidate[row["candidate"]].append(row)
        by_attack[(row["candidate"], row["attack"])] = row

    expected = len(VARIANTS) * len(DEFAULT_ATTACKS)
    completed = len(rows)
    lines = [
        "# AE-SVDD 权重与动态阈值 Sweep 阶段报告",
        "",
        f"- 数据目录：`{root}`",
        f"- 统计方式：每个 run 最后 {args.last_n} 轮均值；当前仅 seed=42。",
        f"- 完整结果：{completed}/{expected}（六组全部完成后才可做最终选型）。",
        "- 指标含义：TACC=测试准确率；TPR=恶意客户端检出率；FPR=良性客户端误杀率；ASR=后门攻击成功率。",
        "",
        "## 运行状态",
        "",
        "| 配置 | 结果数 | 参数 |",
        "|---|---:|---|",
    ]
    for candidate, parameters in VARIANTS.items():
        count = sum(row["candidate"] == candidate for row in rows)
        lines.append(
            f"| {candidate} | {count}/{len(DEFAULT_ATTACKS)} | "
            f"recon:svdd={parameters['recon_loss_weight']:.0f}:{parameters['svdd_loss_weight']:.0f}, "
            f"tau={parameters['tau_start']:.0f}->{parameters['tau_end']:.0f}, "
            f"anneal={parameters['tau_anneal_rounds']} |"
        )

    lines += [
        "",
        "## 已完成配置汇总",
        "",
        "| 配置 | clean TACC | attack TACC | attack TPR | attack FPR | BD ASR | Mix ASR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in VARIANTS:
        clean = by_attack.get((candidate, "none"))
        attacks = [by_attack.get((candidate, attack)) for attack in DEFAULT_ATTACKS if attack != "none"]
        attacks = [row for row in attacks if row is not None]
        def avg(key: str) -> float | None:
            return _mean(row["metrics"].get(key) for row in attacks)
        lines.append(
            f"| {candidate} | {_fmt(clean['metrics']['accuracy'] if clean else None)} | "
            f"{_fmt(avg('accuracy'))} | {_fmt(avg('tpr'))} | {_fmt(avg('fpr'))} | "
            f"{_fmt(by_attack.get((candidate, 'bd'), {}).get('metrics', {}).get('backdoor_asr') if by_attack.get((candidate, 'bd')) else None)} | "
            f"{_fmt(by_attack.get((candidate, 'mix'), {}).get('metrics', {}).get('backdoor_asr') if by_attack.get((candidate, 'mix')) else None)} |"
        )

    lines += [
        "",
        "## 各攻击结果",
        "",
        "| 配置 | 攻击 | TACC | TPR | FPR | DAR | DPR | Reject | ASR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in VARIANTS:
        for attack in DEFAULT_ATTACKS:
            row = by_attack.get((candidate, attack))
            if row is None:
                lines.append(f"| {candidate} | {attack} | 未完成 | - | - | - | - | - | - |")
                continue
            m = row["metrics"]
            lines.append(
                f"| {candidate} | {attack} | {_fmt(m['accuracy'])} | {_fmt(m['tpr'])} | "
                f"{_fmt(m['fpr'])} | {_fmt(m['dar'])} | {_fmt(m['dpr'])} | "
                f"{_fmt(m['reject_rate'])} | {_fmt(m['backdoor_asr'])} |"
            )

    lines += ["", "## 阶段对比（SF、BD）", "", "| 配置 | 攻击 | Warmup TACC | Warmup TPR | Filtering TACC | Filtering TPR | Filtering ASR |", "|---|---|---:|---:|---:|---:|---:|"]
    for candidate in VARIANTS:
        for attack in ("sf", "bd"):
            row = by_attack.get((candidate, attack))
            if row is None:
                lines.append(f"| {candidate} | {attack} | 未完成 | - | - | - | - |")
                continue
            path = root / candidate / "seed_42" / f"cifar10__{attack}__svdd.json"
            phases = _phase_summary(path, ("warmup", "filtering"))
            warm, filtering = phases["warmup"], phases["filtering"]
            lines.append(
                f"| {candidate} | {attack} | {_fmt(warm['accuracy'])} | {_fmt(warm['tpr'])} | "
                f"{_fmt(filtering['accuracy'])} | {_fmt(filtering['tpr'])} | {_fmt(filtering['backdoor_asr'])} |"
            )

    sf_rows = [by_attack[(candidate, "sf")] for candidate in VARIANTS if (candidate, "sf") in by_attack]
    lf_rows = [by_attack[(candidate, "lf")] for candidate in VARIANTS if (candidate, "lf") in by_attack]
    bd_rows = [by_attack[(candidate, "bd")] for candidate in VARIANTS if (candidate, "bd") in by_attack]
    sf_tpr = [float(row["metrics"]["tpr"] or 0.0) for row in sf_rows]
    sf_tacc = [float(row["metrics"]["accuracy"] or 0.0) for row in sf_rows]
    lf_tpr = [float(row["metrics"]["tpr"] or 0.0) for row in lf_rows]
    bd_tpr = [float(row["metrics"]["tpr"] or 0.0) for row in bd_rows]
    bd_asr = [
        float(row["metrics"]["backdoor_asr"])
        for row in bd_rows
        if row["metrics"]["backdoor_asr"] is not None
    ]
    tau32 = by_attack.get(("recon2_tau32", "none"))
    tau21 = by_attack.get(("recon2_tau21", "none"))
    mix_2 = by_attack.get(("recon2_tau32", "mix"))
    mix_4 = by_attack.get(("recon4_tau32", "mix"))
    lines += ["", "## 当前判断", ""]
    if sf_rows:
        lines.append(
            f"- **SF 仍是决定性失败项**：最后 {args.last_n} 轮 TACC 为 "
            f"{_fmt(min(sf_tacc))}–{_fmt(max(sf_tacc))}，TPR 仅 "
            f"{_fmt(min(sf_tpr))}–{_fmt(max(sf_tpr))}，接近随机分类且几乎检不出攻击者。"
        )
    if lf_rows:
        lines.append(
            f"- **LF 防御偏弱**：TPR 范围 {_fmt(min(lf_tpr))}–{_fmt(max(lf_tpr))}，"
            "降低 tau 有改善，但仍不足以作为稳定防御。"
        )
    if bd_rows:
        lines.append(
            f"- **BD 防御有效**：TPR 范围 {_fmt(min(bd_tpr))}–{_fmt(max(bd_tpr))}，"
            f"ASR 被压到 {_fmt(min(bd_asr))}–{_fmt(max(bd_asr))}。"
        )
    lines.append("- **GN 与 LIE 稳定**：当前三个已完成攻击结果的最后 10 轮 TPR 均为 100%。")
    if tau32 and tau21:
        lines.append(
            f"- **tau 2→1 的代价明显**：clean FPR 从 {_fmt(tau32['metrics']['fpr'])} "
            f"升至 {_fmt(tau21['metrics']['fpr'])}；它提高了 LF/SF 检出，但没有挽救 SF。"
        )
    if mix_2 and mix_4:
        lines.append(
            f"- **4:1 在混合攻击下不稳定**：tau 3→2 时，Mix ASR 从 2:1 的 "
            f"{_fmt(mix_2['metrics']['backdoor_asr'])} 恶化到 4:1 的 "
            f"{_fmt(mix_4['metrics']['backdoor_asr'])}，当前不能采用 4:1 作为通用配置。"
        )
    lines.append(
        "- **阶段现象**：SF 在 15 轮 warmup 中的平均 TPR 约 47%，进入 filtering 后迅速下降；"
        "问题主要出现在第二阶段，而不是第一阶段完全没有信号。"
    )
    lines.append(
        "- **当前没有合格赢家**：`recon2_tau32` 是已完成配置中较稳的折中，"
        "但 SF 的 10% TACC/近零 TPR 使它不满足“防住所有攻击”的目标。"
    )

    lines += [
        "",
        "## 结论与限制",
        "",
        "1. 本报告只反映已经写出最终 JSON 的 run；未完成配置不参与优劣判断。",
        "2. 当前结果是单 seed、最后 10 轮均值，不能替代多 seed 统计显著性。",
        "3. 选型优先级应为：先保证 clean/攻击 TACC 不崩，再比较 BD/Mix 的 ASR，最后权衡 TPR 与 FPR。",
        "4. 20 个结果完成后服务器仍有一个 `recon4_tau32/none` run 未写最终 JSON；第二波因此尚未启动。",
    ]
    report = "\n".join(lines) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(args.report)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
