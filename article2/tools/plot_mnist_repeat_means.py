#!/usr/bin/env python3
"""Plot mean TAcc and DAR across the repeated MNIST sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


FACTORS = {
    "lambda": ("SVDD loss ratio $\\lambda$", (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)),
    "phase1": ("Phase-1 rounds", (5.0, 10.0, 15.0, 30.0, 50.0, 100.0)),
    "validation": ("Trusted validation samples", (10.0, 50.0, 100.0, 500.0)),
    "latent": ("Latent dimension", (8.0, 32.0, 64.0, 256.0, 512.0, 4096.0)),
}


def label(factor: str, value: float) -> str:
    return f"{value:.1f}" if factor == "lambda" else str(int(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.summary_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[tuple[str, float], dict[str, list[float]]] = defaultdict(lambda: {"tacc": [], "dar": []})
    for row in rows:
        key = (row["factor"], float(row["factor_value"]))
        grouped[key]["tacc"].append(float(row["final_accuracy_mean"]))
        grouped[key]["dar"].append(float(row["final_dar_mean"]))

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0), sharey=True)
    colors = {"tacc": "#1479b8", "dar": "#e26a2c"}
    markers = {"tacc": "o", "dar": "^"}
    for axis, (factor, (title, expected_values)) in zip(axes.flat, FACTORS.items()):
        positions = list(range(len(expected_values)))
        for metric, metric_label in (("tacc", "TAcc"), ("dar", "DAR")):
            points = []
            for position, value in enumerate(expected_values):
                values = grouped.get((factor, value), {}).get(metric, [])
                if values:
                    points.append((position, statistics.fmean(values)))
            if points:
                axis.plot(
                    [x for x, _ in points], [y for _, y in points],
                    color=colors[metric], marker=markers[metric], linewidth=2.0,
                    markersize=5.5, label=metric_label,
                )
        observed = sum(bool(grouped.get((factor, value))) for value in expected_values)
        repeats = max((len(grouped.get((factor, value), {}).get("tacc", [])) for value in expected_values), default=0)
        axis.set_title(f"{title}\n{observed}/{len(expected_values)} values, max n={repeats}")
        axis.set_xticks(positions)
        axis.set_xticklabels([label(factor, value) for value in expected_values])
        axis.set_xlabel(title)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, linestyle=":", linewidth=1.2, color="#707070", alpha=0.85)
        axis.legend(loc="best", frameon=True)
        for spine in axis.spines.values():
            spine.set_color("#202020")
            spine.set_linewidth(1.0)
    axes[0, 0].set_ylabel("Mean score")
    axes[1, 0].set_ylabel("Mean score")
    fig.suptitle("MNIST repeated sensitivity experiments: mean TAcc and DAR", y=0.98, fontsize=16)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"rows={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
