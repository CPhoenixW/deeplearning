#!/usr/bin/env python3
"""Summarize and plot the one-factor-at-a-time AE-SVDD 5.2 sweep.

Only complete 300-round runs enter the plots.  The summary uses the mean of
the final ten rounds, matching the reporting convention in experiments.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_TASKS = ("mnist", "fashion_mnist", "cifar10", "ag_news", "covid19")
TASK_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "covid19": "COVID-19",
    "ag_news": "AG News",
}
FACTORS = {
    "lambda": ("SVDD loss ratio $\\lambda$", (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)),
    "phase1": ("Phase-1 rounds", (5.0, 10.0, 15.0, 30.0, 50.0, 100.0)),
    "validation": ("Trusted validation samples", (10.0, 50.0, 100.0, 500.0)),
    "latent": ("Latent dimension", (8.0, 32.0, 64.0, 256.0, 512.0, 4096.0)),
}
METRICS = ("accuracy", "balanced_accuracy", "tpr", "fpr", "dar", "dpr", "rr", "reject_rate")


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean_tail(rounds: list[dict[str, Any]], metric: str, last_n: int) -> float | None:
    values = []
    for record in rounds[-last_n:]:
        evaluation = record.get("evaluation", {})
        if isinstance(evaluation, dict):
            value = finite(evaluation.get(metric))
            if value is not None:
                values.append(value)
    return statistics.fmean(values) if values else None


def factor_info(name: str) -> tuple[str, float] | None:
    if name.startswith("lambda_"):
        return "lambda", float(name.removeprefix("lambda_").replace("p", "."))
    if name.startswith("phase1_"):
        return "phase1", float(name.removeprefix("phase1_"))
    if name.startswith("validation_"):
        return "validation", float(name.removeprefix("validation_"))
    if name.startswith("latent_"):
        return "latent", float(name.removeprefix("latent_"))
    return None


def scan(input_root: Path, expected_rounds: int, last_n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for path in sorted(input_root.rglob("*__gn__svdd.json")):
        if "_configs" in path.parts:
            continue
        info = factor_info(path.parent.parent.name)
        if info is None:
            incomplete.append({"path": str(path), "reason": "unknown_factor"})
            continue
        factor, value = info
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            incomplete.append({"path": str(path), "reason": f"parse_error:{type(exc).__name__}"})
            continue
        rounds = payload.get("rounds") if isinstance(payload, dict) else None
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if not isinstance(rounds, list) or len(rounds) != expected_rounds:
            incomplete.append({"path": str(path), "reason": f"rounds:{len(rounds) if isinstance(rounds, list) else 0}/{expected_rounds}"})
            continue
        task = str(meta.get("task", path.parts[-4]))
        row: dict[str, Any] = {
            "task": task,
            "factor": factor,
            "factor_value": value,
            "rounds": len(rounds),
            "path": str(path),
        }
        for metric in METRICS:
            row[f"final_{metric}_mean"] = mean_tail(rounds, metric, last_n)
        complete.append(row)
    return complete, incomplete


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = {"factor_value", "rounds", *(f"final_{metric}_mean" for metric in METRICS)}
    for row in rows:
        for key in numeric:
            value = row.get(key)
            if value in (None, ""):
                row[key] = None
            else:
                row[key] = float(value)
        row["rounds"] = int(row["rounds"])
    return rows


def plot_factor(rows: list[dict[str, Any]], factor: str, output: Path, tasks: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt

    label, expected_values = FACTORS[factor]
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, axes = plt.subplots(2, len(tasks), figsize=(max(16.2, 3.5 * len(tasks)), 6.7), sharey="row")
    colors = {"accuracy": "#1479b8", "dar": "#e26a2c"}
    markers = {"accuracy": "o", "dar": "^"}
    for column, task in enumerate(tasks):
        task_rows = sorted(
            (row for row in rows if row["task"] == task and row["factor"] == factor),
            key=lambda row: row["factor_value"],
        )
        complete_values = {float(row["factor_value"]): row for row in task_rows}
        x_positions = list(range(len(expected_values)))
        for row_index, metric in enumerate(("accuracy", "dar")):
            axis = axes[row_index, column]
            points = [
                (position, complete_values[value])
                for position, value in enumerate(expected_values)
                if value in complete_values and complete_values[value].get(f"final_{metric}_mean") is not None
            ]
            if points:
                axis.plot(
                    [position for position, _ in points],
                    [float(row[f"final_{metric}_mean"]) for _, row in points],
                    color=colors[metric], marker=markers[metric], linewidth=2.0, markersize=5.5,
                )
            missing = len(expected_values) - len(task_rows)
            if missing:
                axis.text(
                    0.5, 0.5, f"pending\n{len(task_rows)}/{len(expected_values)} complete",
                    transform=axis.transAxes, ha="center", va="center", color="#777777", fontsize=10,
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f2f2f2", "edgecolor": "#b0b0b0", "alpha": 0.9},
                )
            # Treat swept values as categorical levels so irregular numeric gaps
            # (e.g. 15 -> 50 -> 100 or 64 -> 512 -> 4096) remain equally spaced.
            axis.set_xticks(x_positions)
            axis.set_ylim(0.0, 1.05)
            axis.grid(True, linestyle=":", linewidth=1.2, color="#707070", alpha=0.85)
            for spine in axis.spines.values():
                spine.set_color("#202020")
                spine.set_linewidth(1.0)
            if row_index == 0:
                axis.set_title(f"({chr(97 + column)}) {TASK_LABELS[task]}\n$n={len(task_rows)}/{len(expected_values)}$")
            if row_index == 1:
                axis.set_xlabel(label)
            if column == 0:
                axis.set_ylabel("Test accuracy (TAcc)" if row_index == 0 else "Detection accuracy (DAR)")
            if factor == "lambda":
                axis.set_xticklabels([f"{v:.1f}" for v in expected_values])
            elif factor == "phase1":
                axis.set_xticklabels([str(int(v)) for v in expected_values])
            else:
                axis.set_xticklabels([str(int(v)) for v in expected_values])
    fig.suptitle(f"AE-SVDD sensitivity to {label}", y=1.01, fontsize=15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rounds", type=int, default=300)
    parser.add_argument("--last-n", type=int, default=10)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Plot an existing compact summary instead of scanning JSON files.")
    args = parser.parse_args()
    tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    unknown_tasks = sorted(set(tasks) - set(TASK_LABELS))
    if not tasks or unknown_tasks:
        parser.error(f"tasks must be selected from {tuple(TASK_LABELS)}")
    if args.summary_csv is not None:
        complete = read_summary_csv(args.summary_csv.resolve())
        incomplete = []
    else:
        complete, incomplete = scan(args.input_root.resolve(), args.expected_rounds, args.last_n)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_by_run.csv", complete)
    write_csv(args.output_dir / "incomplete.csv", incomplete)
    report = {
        "expected_runs": len(tasks) * sum(len(values) for _, values in FACTORS.values()),
        "complete_runs": len(complete),
        "incomplete_or_missing": len(tasks) * sum(len(values) for _, values in FACTORS.values()) - len(complete),
        "last_n_rounds": args.last_n,
        "expected_rounds": args.expected_rounds,
        "complete_by_task": {task: sum(row["task"] == task for row in complete) for task in tasks},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.no_plots:
        for factor in FACTORS:
            plot_factor(complete, factor, args.output_dir / f"svdd-52-{factor}.png", tasks)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
