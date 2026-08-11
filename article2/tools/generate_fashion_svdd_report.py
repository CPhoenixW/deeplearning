#!/usr/bin/env python3
"""Generate an auditable multi-table report for a Fashion-MNIST SVDD matrix.

The matrix runner deliberately writes one JSON file per independent
``(phase1, malicious ratio, score mode, seed, attack)`` run.  This tool turns
those files into two complementary artifacts:

* ``fashion_svdd_report.md``: readable tables for protocol, coverage,
  sensitivity, attack-by-attack results, phase transitions, and stability.
* ``fashion_svdd_all_runs.csv``: one row per expected run, including pending
  and running jobs, all final/last-10/phase-1 metrics, and diagnostics.

No missing result is silently dropped or averaged as zero.  A pending cell is
rendered explicitly, which makes the in-progress report safe to update while
the scheduler is still running.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ATTACK_ORDER = ("none", "gn", "lf", "sf", "bd", "lie", "minmax", "minsum", "mix")
MODE_ORDER = ("recon", "combined", "svdd", "legacy")
ERROR_MARKERS = (
    "Traceback",
    "FloatingPointError",
    "All SVDD feature rows are non-finite",
    "CUDA out of memory",
    "OOM",
    "Killed",
)
EVAL_METRICS = (
    "accuracy",
    "tpr",
    "fpr",
    "dar",
    "dpr",
    "rr",
    "reject_rate",
    "backdoor_asr",
)
SERVER_METRICS = (
    "selected_reject_ratio",
    "validation_accuracy",
    "diagnostic_upload_clip_fraction",
    "diagnostic_upload_clipped_count",
    "diagnostic_upload_nonfinite_replaced_count",
    "diagnostic_upload_update_norm_median",
    "diagnostic_upload_update_norm_max",
    "global_state_rolled_back",
    "svdd_loss",
    "recon_loss",
    "total_loss",
)
ACTIVE_CONFIG_KEYS = (
    "num_clients",
    "total_rounds",
    "client_lr",
    "client_momentum",
    "client_weight_decay",
    "local_epochs",
    "batch_size",
    "dirichlet_alpha",
    "client_grad_clip",
    "client_update_clip",
    "latent_dim",
    "ae_lr",
    "ae_weight_decay",
    "ae_grad_clip",
    "svdd_input_mode",
    "svdd_feature_mode",
    "param_descriptor_dim",
    "param_descriptor_global_ratio",
    "param_descriptor_layer_ratio",
    "param_descriptor_statistics_ratio",
    "param_descriptor_seed",
    "param_descriptor_device",
    "center_ema_decay",
    "svdd_grad_clip",
    "svdd_feature_clip",
    "center_init_quantile",
    "phase2_recon_quantile",
    "backdoor_target_label",
    "backdoor_poison_ratio",
    "backdoor_trigger_size",
    "backdoor_trigger_value",
    "backdoor_model_replace_scale",
    "gaussian_sigma",
    "sign_flip_scale",
    "lie_z_override",
    "distance_attack_deviation",
    "mixed_attack_types",
)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def _std(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if len(finite) < 2:
        return 0.0 if finite else None
    return statistics.stdev(finite)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f}%"


def _fmt_mean_std(values: Iterable[float | None], *, percent: bool = True) -> str:
    mean = _mean(values)
    std = _std(values)
    if mean is None:
        return "—"
    if percent:
        return f"{_fmt_pct(mean)} ± {_fmt_pct(std)}"
    return f"{_fmt(mean)} ± {_fmt(std)}"


def _fmt_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _last_round_in_jsonl(paths: Sequence[Path]) -> int:
    """Read only the tail of logs, so progress reporting stays lightweight."""

    last_round = 0
    for path in paths:
        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - 131_072))
                tail = stream.read().decode("utf-8", "replace")
        except OSError:
            continue
        for line in reversed(tail.splitlines()):
            if '"record_type":"round"' not in line:
                continue
            try:
                payload = json.loads(line)
                last_round = max(last_round, int(payload.get("round", 0)))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            break
    return last_round


def _round_metric(round_payload: Mapping[str, Any], section: str, key: str) -> float | None:
    section_payload = round_payload.get(section)
    if not isinstance(section_payload, Mapping):
        return None
    return _numeric(section_payload.get(key))


def _metric_series(rounds: Sequence[Mapping[str, Any]], section: str, key: str) -> list[float | None]:
    return [_round_metric(round_payload, section, key) for round_payload in rounds]


def _phase_metric(
    rounds: Sequence[Mapping[str, Any]],
    phase1_rounds: int,
    section: str,
    key: str,
) -> float | None:
    if not rounds:
        return None
    index = min(max(0, int(phase1_rounds) - 1), len(rounds) - 1)
    return _round_metric(rounds[index], section, key)


def _extract_metrics(record: dict[str, Any], rounds: Sequence[Mapping[str, Any]]) -> None:
    if not rounds:
        return
    final = rounds[-1]
    last_ten = rounds[-10:]
    phase1 = int(record["phase1_rounds"])
    for key in EVAL_METRICS:
        values = _metric_series(rounds, "evaluation", key)
        record[f"final_{key}"] = values[-1]
        record[f"last10_{key}"] = _mean(values[-10:])
        record[f"phase1_{key}"] = _phase_metric(rounds, phase1, "evaluation", key)
    for key in SERVER_METRICS:
        values = _metric_series(rounds, "server", key)
        record[f"final_{key}"] = values[-1]
        record[f"last10_{key}"] = _mean(values[-10:])
        record[f"phase1_{key}"] = _phase_metric(rounds, phase1, "server", key)
    record["last_round_phase"] = str(final.get("phase", ""))


def _config_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    attack = str(record["attack"])
    mode = str(record["score_mode"])
    return (
        ATTACK_ORDER.index(attack) if attack in ATTACK_ORDER else len(ATTACK_ORDER),
        int(record["phase1_rounds"]),
        int(record["num_malicious"]),
        MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER),
        int(record["seed"]),
    )


def _load_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs_root = root / "_configs"
    config_paths = sorted(configs_root.rglob("*.json"))
    records: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for config_path in config_paths:
        payload = _read_json(config_path)
        if payload is None:
            malformed.append({"path": str(config_path), "reason": "invalid config JSON"})
            continue
        overrides = payload.get("fed_config_overrides", {})
        if not isinstance(overrides, Mapping):
            malformed.append({"path": str(config_path), "reason": "missing fed_config_overrides"})
            continue
        try:
            attack = str(payload["attacks"])
            task = str(payload.get("task", "fashion_mnist"))
            output_dir = Path(str(payload["log_dir"]))
            total_rounds = int(overrides["total_rounds"])
            phase1_rounds = int(overrides["phase1_rounds"])
            num_clients = int(overrides["num_clients"])
            configured_num_malicious = int(overrides["num_malicious"])
            seed = int(overrides["seed"])
            score_mode = str(overrides["svdd_score_mode"])
            alpha = float(overrides["alpha"])
        except (KeyError, TypeError, ValueError) as exc:
            malformed.append({"path": str(config_path), "reason": f"invalid config fields: {exc}"})
            continue

        # The scheduler keeps one clean control at a reference matrix ratio so
        # its config path is deterministic.  The pipeline itself creates no
        # malicious clients for ``attack=none``; report its *actual* attack
        # cardinality as zero instead of the unused reference value.
        num_malicious = 0 if attack == "none" else configured_num_malicious
        result_path = output_dir / f"{task}__{attack}__svdd.json"
        jsonl_paths = sorted(output_dir.glob(f"{task}__{attack}__svdd__*.jsonl"))
        console_path = output_dir / f"{attack}.log"
        record: dict[str, Any] = {
            "experiment_id": (
                f"p1={phase1_rounds};M={num_malicious};mode={score_mode};"
                f"seed={seed};attack={attack}"
            ),
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "result_path": str(result_path),
            "task": task,
            "attack": attack,
            "score_mode": score_mode,
            "alpha": alpha,
            "seed": seed,
            "phase1_rounds": phase1_rounds,
            "total_rounds": total_rounds,
            "num_clients": num_clients,
            "num_malicious": num_malicious,
            "malicious_ratio": num_malicious / max(1, num_clients),
            "current_round": _last_round_in_jsonl(jsonl_paths),
            "status": "pending",
            "error_marker": "",
            "effective_config": None,
            "overrides": dict(overrides),
        }
        result = _read_json(result_path) if result_path.exists() else None
        if result is not None:
            rounds = result.get("rounds")
            if isinstance(rounds, list):
                record["current_round"] = max(record["current_round"], len(rounds))
                meta = result.get("meta", {})
                if isinstance(meta, Mapping) and isinstance(meta.get("effective_config"), Mapping):
                    record["effective_config"] = dict(meta["effective_config"])
                if len(rounds) == total_rounds:
                    record["status"] = "completed"
                    typed_rounds = [
                        round_payload
                        for round_payload in rounds
                        if isinstance(round_payload, Mapping)
                    ]
                    _extract_metrics(record, typed_rounds)
                else:
                    record["status"] = "partial"
            else:
                record["status"] = "unreadable"
                record["error_marker"] = "result JSON has no rounds array"
        elif jsonl_paths:
            record["status"] = "running" if record["current_round"] > 0 else "started"

        if console_path.exists():
            try:
                console_text = console_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                console_text = ""
            markers = [marker for marker in ERROR_MARKERS if marker in console_text]
            if markers:
                record["status"] = "failed"
                record["error_marker"] = "; ".join(markers)
        records.append(record)
    return sorted(records, key=_config_sort_key), malformed


def _seed_cell(record: Mapping[str, Any] | None, key: str, *, percent: bool = True) -> str:
    if record is None:
        return "pending"
    if record.get("status") != "completed":
        round_index = int(record.get("current_round", 0))
        status = str(record.get("status", "pending"))
        return f"{status} r{round_index}" if round_index else status
    value = _numeric(record.get(key))
    return _fmt_pct(value) if percent else _fmt(value)


def _group_values(records: Sequence[Mapping[str, Any]], key: str) -> list[float | None]:
    return [_numeric(record.get(key)) for record in records if record.get("status") == "completed"]


def _result_summary_rows(
    records: Sequence[Mapping[str, Any]], group_keys: Sequence[str]
) -> list[list[str]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in group_keys)].append(record)
    rows: list[list[str]] = []
    for group in sorted(groups):
        group_records = groups[group]
        completed = [record for record in group_records if record["status"] == "completed"]
        row = [_fmt_value(value) for value in group]
        row.extend(
            [
                f"{len(completed)}/{len(group_records)}",
                _fmt_mean_std(_group_values(group_records, "final_accuracy")),
                _fmt_mean_std(_group_values(group_records, "final_tpr")),
                _fmt_mean_std(_group_values(group_records, "final_fpr")),
                _fmt_mean_std(_group_values(group_records, "final_backdoor_asr")),
                _fmt_mean_std(_group_values(group_records, "last10_accuracy")),
            ]
        )
        rows.append(row)
    return rows


def _mode_label(mode: str, alpha: float) -> str:
    return f"{mode} (α={alpha:g})"


def _attack_detail_tables(records: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> str:
    chunks: list[str] = []
    for attack in ATTACK_ORDER:
        attack_records = [record for record in records if record["attack"] == attack]
        if not attack_records:
            continue
        chunks.append(f"## 攻击明细：{attack.upper()}")
        grouped: dict[tuple[int, int, str, float], dict[int, Mapping[str, Any]]] = defaultdict(dict)
        for record in attack_records:
            grouped[
                (
                    int(record["phase1_rounds"]),
                    int(record["num_malicious"]),
                    str(record["score_mode"]),
                    float(record["alpha"]),
                )
            ][int(record["seed"])] = record

        headers = ["P1", "恶意数", "恶意比例", "模式"]
        headers.extend(f"seed {seed}" for seed in seeds)
        headers.extend(["TACC 均值 ± SD", "完成"])
        tacc_rows: list[list[str]] = []
        for (p1, malicious, mode, alpha), by_seed in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1], MODE_ORDER.index(item[0][2]))
        ):
            ordered = [by_seed.get(seed) for seed in seeds]
            completed = [record for record in ordered if record and record["status"] == "completed"]
            tacc_rows.append(
                [
                    str(p1),
                    str(malicious),
                    _fmt_pct(malicious / 100.0, 0),
                    _mode_label(mode, alpha),
                    *[_seed_cell(record, "final_accuracy") for record in ordered],
                    _fmt_mean_std([_numeric(record.get("final_accuracy")) for record in completed]),
                    f"{len(completed)}/{len(seeds)}",
                ]
            )
        chunks.append("### 最终轮 TACC（每个种子单列）\n\n" + _table(headers, tacc_rows))

        detection_headers = ["P1", "恶意数", "模式"]
        for seed in seeds:
            detection_headers.extend([f"S{seed} TPR", f"S{seed} FPR"])
        detection_headers.extend(["TPR 均值 ± SD", "FPR 均值 ± SD", "完成"])
        detection_rows: list[list[str]] = []
        for (p1, malicious, mode, alpha), by_seed in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1], MODE_ORDER.index(item[0][2]))
        ):
            ordered = [by_seed.get(seed) for seed in seeds]
            completed = [record for record in ordered if record and record["status"] == "completed"]
            seed_metrics: list[str] = []
            for record in ordered:
                seed_metrics.extend(
                    [_seed_cell(record, "final_tpr"), _seed_cell(record, "final_fpr")]
                )
            detection_rows.append(
                [
                    str(p1),
                    str(malicious),
                    _mode_label(mode, alpha),
                    *seed_metrics,
                    _fmt_mean_std([_numeric(record.get("final_tpr")) for record in completed]),
                    _fmt_mean_std([_numeric(record.get("final_fpr")) for record in completed]),
                    f"{len(completed)}/{len(seeds)}",
                ]
            )
        chunks.append("### 最终轮参与方检出（TPR / FPR）\n\n" + _table(detection_headers, detection_rows))

        if attack in {"bd", "mix"}:
            asr_headers = ["P1", "恶意数", "模式"]
            asr_headers.extend(f"seed {seed} ASR" for seed in seeds)
            asr_headers.extend(["ASR 均值 ± SD", "完成"])
            asr_rows: list[list[str]] = []
            for (p1, malicious, mode, alpha), by_seed in sorted(
                grouped.items(), key=lambda item: (item[0][0], item[0][1], MODE_ORDER.index(item[0][2]))
            ):
                ordered = [by_seed.get(seed) for seed in seeds]
                completed = [record for record in ordered if record and record["status"] == "completed"]
                asr_rows.append(
                    [
                        str(p1),
                        str(malicious),
                        _mode_label(mode, alpha),
                        *[_seed_cell(record, "final_backdoor_asr") for record in ordered],
                        _fmt_mean_std(
                            [_numeric(record.get("final_backdoor_asr")) for record in completed]
                        ),
                        f"{len(completed)}/{len(seeds)}",
                    ]
                )
            chunks.append("### 最终轮后门 ASR（越低越好）\n\n" + _table(asr_headers, asr_rows))
    return "\n\n".join(chunks)


def _phase_transition_table(records: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "攻击",
        "模式",
        "完成",
        "P1末 TACC",
        "最终 TACC",
        "P1末 TPR",
        "最终 TPR",
        "P1末 FPR",
        "最终 FPR",
    ]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["attack"]), str(record["score_mode"]))].append(record)
    rows: list[list[str]] = []
    for (attack, mode), group_records in sorted(
        groups.items(),
        key=lambda item: (
            ATTACK_ORDER.index(item[0][0]) if item[0][0] in ATTACK_ORDER else 999,
            MODE_ORDER.index(item[0][1]) if item[0][1] in MODE_ORDER else 999,
        ),
    ):
        completed = [record for record in group_records if record["status"] == "completed"]
        rows.append(
            [
                attack.upper(),
                mode,
                f"{len(completed)}/{len(group_records)}",
                _fmt_mean_std(_group_values(group_records, "phase1_accuracy")),
                _fmt_mean_std(_group_values(group_records, "final_accuracy")),
                _fmt_mean_std(_group_values(group_records, "phase1_tpr")),
                _fmt_mean_std(_group_values(group_records, "final_tpr")),
                _fmt_mean_std(_group_values(group_records, "phase1_fpr")),
                _fmt_mean_std(_group_values(group_records, "final_fpr")),
            ]
        )
    return _table(headers, rows)


def _selection_stability_table(records: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "攻击",
        "模式",
        "完成",
        "最终拒绝比例",
        "最后10轮拒绝比例",
        "最后10轮验证准确率",
        "最后10轮上传裁剪比例",
        "最后10轮非有限替换数",
        "最后10轮全局回滚",
    ]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["attack"]), str(record["score_mode"]))].append(record)
    rows: list[list[str]] = []
    for (attack, mode), group_records in sorted(
        groups.items(),
        key=lambda item: (
            ATTACK_ORDER.index(item[0][0]) if item[0][0] in ATTACK_ORDER else 999,
            MODE_ORDER.index(item[0][1]) if item[0][1] in MODE_ORDER else 999,
        ),
    ):
        completed = [record for record in group_records if record["status"] == "completed"]
        rows.append(
            [
                attack.upper(),
                mode,
                f"{len(completed)}/{len(group_records)}",
                _fmt_mean_std(
                    _group_values(group_records, "final_selected_reject_ratio")
                ),
                _fmt_mean_std(
                    _group_values(group_records, "last10_selected_reject_ratio")
                ),
                _fmt_mean_std(_group_values(group_records, "last10_validation_accuracy")),
                _fmt_mean_std(
                    _group_values(group_records, "last10_diagnostic_upload_clip_fraction")
                ),
                _fmt_mean_std(
                    _group_values(
                        group_records, "last10_diagnostic_upload_nonfinite_replaced_count"
                    ),
                    percent=False,
                ),
                _fmt_mean_std(
                    _group_values(group_records, "last10_global_state_rolled_back"),
                    percent=False,
                ),
            ]
        )
    return _table(headers, rows)


def _failure_table(records: Sequence[Mapping[str, Any]], malformed: Sequence[Mapping[str, Any]]) -> str:
    rows: list[list[str]] = []
    for record in records:
        if record["status"] in {"failed", "unreadable", "partial"}:
            rows.append(
                [
                    str(record["status"]),
                    str(record["attack"]),
                    str(record["phase1_rounds"]),
                    str(record["num_malicious"]),
                    str(record["score_mode"]),
                    str(record["seed"]),
                    str(record.get("error_marker", "")),
                    str(record["output_dir"]),
                ]
            )
    for item in malformed:
        rows.append(["malformed", "—", "—", "—", "—", "—", str(item["reason"]), str(item["path"])])
    if not rows:
        rows.append(["none", "—", "—", "—", "—", "—", "—", "—"])
    return _table(["状态", "攻击", "P1", "恶意数", "模式", "seed", "原因", "路径"], rows)


def _report_markdown(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    malformed: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    status_counts = Counter(str(record["status"]) for record in records)
    completed = status_counts["completed"]
    expected = len(records)
    seeds = sorted({int(record["seed"]) for record in records})
    sample_config = next(
        (
            record["effective_config"]
            for record in records
            if isinstance(record.get("effective_config"), Mapping)
        ),
        None,
    )
    if sample_config is None and records:
        sample_config = records[0].get("overrides", {})
    sample_config = sample_config or {}

    markdown: list[str] = [
        "# Fashion-MNIST AE-SVDD 敏感性实验报告",
        "",
        f"- 报告生成时间：{now}",
        f"- 结果根目录：`{root}`",
        f"- 覆盖度：`{completed}/{expected}` 已完成；其余状态会保留在所有明细表和 CSV 中。",
        "- 指标口径：`final_*` 为第 100 轮；`last10_*` 为最后 10 轮均值；TACC 是全局测试准确率。",
        "- TPR/FPR 为客户端恶意检测率/良性误杀率；BD 与 mix 的 ASR 越低越好。",
        "",
        "## 1. 实验矩阵与执行状态",
        "",
    ]
    status_headers = ["攻击", "recon", "combined", "svdd", "总计"]
    status_rows: list[list[str]] = []
    for attack in ATTACK_ORDER:
        attack_records = [record for record in records if record["attack"] == attack]
        if not attack_records:
            continue
        row = [attack.upper()]
        for mode in ("recon", "combined", "svdd"):
            values = [record for record in attack_records if record["score_mode"] == mode]
            done = sum(record["status"] == "completed" for record in values)
            running = sum(record["status"] in {"running", "started"} for record in values)
            row.append(f"{done}/{len(values)}（运行 {running}）")
        done = sum(record["status"] == "completed" for record in attack_records)
        row.append(f"{done}/{len(attack_records)}")
        status_rows.append(row)
    markdown.append(_table(status_headers, status_rows))
    markdown.extend(
        [
            "",
            "| 状态 | 数量 |",
            "| --- | ---: |",
            *[f"| {status} | {count} |" for status, count in sorted(status_counts.items())],
            "",
            "## 2. 参数设定",
            "",
            "### 2.1 固定训练与防御参数",
            "",
        ]
    )
    config_rows = [
        [key, _fmt_value(sample_config.get(key))]
        for key in ACTIVE_CONFIG_KEYS
        if key in sample_config
    ]
    markdown.append(_table(["参数", "值"], config_rows))
    markdown.extend(
        [
            "",
            "### 2.2 搜索维度",
            "",
            _table(
                ["维度", "取值", "说明"],
                [
                    ["phase1_rounds", "5, 15, 30, 50", "第一阶段持续轮数"],
                    ["恶意参与方", "10, 20, 30, 40 / 100", "每轮固定参与的恶意客户端数"],
                    ["score_mode / alpha", "recon / 0.0；combined / 0.5；svdd / 1.0", "两阶段均使用该筛选分数"],
                    ["seed", ", ".join(map(str, seeds)), "独立重复"],
                    [
                        "攻击",
                        "none, gn, lf, sf, bd, lie, minmax, minsum, mix",
                        "mix=lf,bd,gn,sf,lie,minmax,minsum",
                    ],
                    ["总轮数", str(next(iter({int(record["total_rounds"]) for record in records}), "—")), "每个独立任务"],
                ],
            ),
            "",
            "### 2.3 动态 Top-K 规则",
            "",
            "每轮将候选拒绝比例 `10%/20%/30%/40%/50%` 分别聚合，使用固定干净服务器验证集选取验证准确率最高者；平局选较大的拒绝比例。该比例是逐轮决策结果，不作为额外搜索超参数。",
            "",
            "## 3. 总体结果表（所有已完成任务的分组均值）",
            "",
            "### 3.1 按攻击",
            "",
            _table(
                ["攻击", "完成", "最终 TACC", "最终 TPR", "最终 FPR", "最终 ASR", "最后10轮 TACC"],
                _result_summary_rows(records, ("attack",)),
            ),
            "",
            "### 3.2 按评分模式",
            "",
            _table(
                ["模式", "完成", "最终 TACC", "最终 TPR", "最终 FPR", "最终 ASR", "最后10轮 TACC"],
                _result_summary_rows(records, ("score_mode",)),
            ),
            "",
            "### 3.3 按第一阶段长度与评分模式",
            "",
            _table(
                ["P1", "模式", "完成", "最终 TACC", "最终 TPR", "最终 FPR", "最终 ASR", "最后10轮 TACC"],
                _result_summary_rows(records, ("phase1_rounds", "score_mode")),
            ),
            "",
            "### 3.4 按恶意比例与评分模式",
            "",
            _table(
                ["恶意数", "模式", "完成", "最终 TACC", "最终 TPR", "最终 FPR", "最终 ASR", "最后10轮 TACC"],
                _result_summary_rows(records, ("num_malicious", "score_mode")),
            ),
            "",
            "## 4. 阶段转换：第一阶段末 vs 最终轮",
            "",
            _phase_transition_table(records),
            "",
            "## 5. 动态筛选与数值稳定性",
            "",
            _selection_stability_table(records),
            "",
            "- `上传裁剪比例` 是最后 10 轮中被更新范数边界裁剪的参与方比例。",
            "- `非有限替换数` 和 `全局回滚` 的理想值均为 0。",
            "",
            _attack_detail_tables(records, seeds),
            "",
            "## 6. 失败、不完整或无法读取的任务",
            "",
            _failure_table(records, malformed),
            "",
            "## 7. 可审计明细文件",
            "",
            "- `fashion_svdd_all_runs.csv`：每个预期任务一行，包含参数、状态、最终轮、最后10轮、第一阶段末、动态 Top-K 与裁剪诊断。",
            "- `fashion_svdd_effective_config.json`：从实际完成任务写出的完整 effective config 快照。",
            "- 所有 aggregate 表只聚合 `completed` 行；未完成任务会保留为 pending/running，不参与均值。",
            "",
        ]
    )
    return "\n".join(markdown)


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in {"effective_config", "overrides"}}
        row["effective_config_json"] = json.dumps(record.get("effective_config"), sort_keys=True)
        row["overrides_json"] = json.dumps(record.get("overrides"), sort_keys=True)
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Matrix output root containing _configs/.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for report artifacts (default: <root>/report).",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or (root / "report")).resolve()
    if not (root / "_configs").is_dir():
        parser.error(f"No _configs directory found under {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records, malformed = _load_records(root)
    if not records:
        parser.error(f"No JSON configurations found under {root / '_configs'}")

    report_path = output_dir / "fashion_svdd_report.md"
    csv_path = output_dir / "fashion_svdd_all_runs.csv"
    config_path = output_dir / "fashion_svdd_effective_config.json"
    report_path.write_text(_report_markdown(root, records, malformed, output_dir), encoding="utf-8")
    _write_csv(csv_path, records)
    effective = next(
        (
            record["effective_config"]
            for record in records
            if isinstance(record.get("effective_config"), Mapping)
        ),
        None,
    )
    config_path.write_text(
        json.dumps(effective or {}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_counts = Counter(str(record["status"]) for record in records)
    print(
        json.dumps(
            {
                "expected": len(records),
                "status": dict(sorted(status_counts.items())),
                "report": str(report_path),
                "csv": str(csv_path),
                "effective_config": str(config_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
