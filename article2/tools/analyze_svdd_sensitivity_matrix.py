#!/usr/bin/env python3
"""Summarize completed SVDD sensitivity-matrix JSON results.

The scanner is intentionally dependency-free so it can run on the training
server while the matrix is still producing files. Incomplete or malformed
results are reported separately and never enter the aggregate statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "tacc",
    "tpr",
    "fpr",
    "dar",
    "dpr",
    "rr",
    "reject_rate",
    "backdoor_asr",
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _std(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None)


def _round_metric(record: dict[str, Any], name: str) -> float | None:
    evaluation = record.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return None
    if name == "tacc":
        return _finite(evaluation.get("accuracy"))
    return _finite(evaluation.get(name))


def _factor_from_meta(meta: dict[str, Any]) -> tuple[int | None, float | None, str]:
    effective = meta.get("effective_config", {})
    if not isinstance(effective, dict):
        effective = {}
    p1 = effective.get("phase1_rounds")
    try:
        p1 = int(p1)
    except (TypeError, ValueError):
        p1 = None
    try:
        malicious = int(effective.get("num_clients", 0)) - int(
            effective.get("num_benign", 0)
        )
        ratio = malicious / int(effective.get("num_clients", 1))
    except (TypeError, ValueError, ZeroDivisionError):
        ratio = None
    mode = str(effective.get("svdd_score_mode", ""))
    return p1, ratio, mode


def _load(path: Path, expected_rounds: int, last_n: int) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"parse_error:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "root_not_object"
    meta = payload.get("meta")
    rounds = payload.get("rounds")
    if not isinstance(meta, dict) or not isinstance(rounds, list):
        return None, "missing_meta_or_rounds"
    if len(rounds) != expected_rounds:
        return None, f"incomplete_rounds:{len(rounds)}/{expected_rounds}"
    if not rounds:
        return None, "empty_rounds"
    attack = str(meta.get("attack", path.name.split("__")[1] if "__" in path.name else ""))
    p1, ratio, mode = _factor_from_meta(meta)
    effective = meta.get("effective_config", {})
    if not isinstance(effective, dict):
        effective = {}
    seed = _finite(effective.get("seed"))
    row: dict[str, Any] = {
        "path": str(path),
        "task": str(meta.get("task", "")),
        "attack": attack,
        "defense": str(meta.get("defense", "")),
        "seed": int(seed) if seed is not None else None,
        "phase1_rounds": p1,
        "malicious_ratio": ratio,
        "mode": mode,
        "rounds": len(rounds),
    }
    tail = rounds[-max(1, int(last_n)) :]
    for metric in METRICS:
        values = [_round_metric(record, metric) for record in tail if isinstance(record, dict)]
        row[f"final_{metric}_mean"] = _mean(values)
        row[f"final_{metric}_std"] = _std(values)
    for phase_name, phase_records in (
        ("phase1", [record for record in rounds if isinstance(record, dict) and str(record.get("phase")) == "warmup"]),
        ("phase2", [record for record in rounds if isinstance(record, dict) and str(record.get("phase")) != "warmup"]),
    ):
        phase_tail = phase_records[-max(1, int(last_n)) :]
        for metric in METRICS:
            row[f"{phase_name}_{metric}_mean"] = _mean(
                _round_metric(record, metric) for record in phase_tail
            )
    return row, None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    result: list[dict[str, Any]] = []
    for group_key, members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        output = {key: value for key, value in zip(keys, group_key)}
        output["n_runs"] = len(members)
        for metric in METRICS:
            values = [member.get(f"final_{metric}_mean") for member in members]
            output[f"{metric}_mean"] = _mean(values)
            output[f"{metric}_seed_std"] = _std(values)
        result.append(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-rounds", type=int, default=100)
    parser.add_argument("--last-n", type=int, default=10)
    args = parser.parse_args()
    if args.expected_rounds < 1 or args.last_n < 1:
        parser.error("expected-rounds and last-n must be positive")
    root = args.input_root.resolve()
    output = (args.output_dir or (root / "analysis")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if not path.name.endswith("__svdd.json"):
            continue
        row, error = _load(path, int(args.expected_rounds), int(args.last_n))
        if row is None:
            invalid.append({"path": str(path), "error": error})
        else:
            rows.append(row)
    _write_csv(output / "summary_by_run.csv", rows)
    _write_csv(output / "invalid_or_incomplete.csv", invalid)
    by_attack = _aggregate(
        rows, ("task", "phase1_rounds", "malicious_ratio", "mode", "attack")
    )
    by_factor = _aggregate(rows, ("task", "phase1_rounds", "malicious_ratio", "mode"))
    _write_csv(output / "summary_by_attack.csv", by_attack)
    _write_csv(output / "summary_by_factor.csv", by_factor)
    report = {
        "input_root": str(root),
        "expected_rounds": int(args.expected_rounds),
        "last_n": int(args.last_n),
        "complete_runs": len(rows),
        "invalid_or_incomplete": len(invalid),
        "expected_files": len(rows) + len(invalid),
        "outputs": {
            "summary_by_run": str(output / "summary_by_run.csv"),
            "summary_by_attack": str(output / "summary_by_attack.csv"),
            "summary_by_factor": str(output / "summary_by_factor.csv"),
            "invalid_or_incomplete": str(output / "invalid_or_incomplete.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
