#!/usr/bin/env python3
"""Plot Fashion-MNIST GN/SF/BD SVDD sensitivity means across seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ATTACKS = ("gn", "sf", "bd")
ATTACK_LABELS = {"gn": "GN", "sf": "SF", "bd": "BD"}
FACTORS = {
    "lambda": ("SVDD loss ratio $\\lambda$", (0.01, 0.1, 0.5, 0.9, 0.99)),
    "phase1": ("Phase-1 rounds", (5.0, 15.0, 30.0, 50.0, 100.0)),
    "validation": ("Trusted validation samples", (10.0, 20.0, 50.0, 100.0, 200.0)),
    "latent": ("Latent dimension", (8.0, 32.0, 64.0, 128.0, 256.0)),
}


def factor_info(name: str) -> tuple[str, float]:
    factor, raw = name.split("_", 1)
    return factor, float(raw.replace("p", "."))


def value_label(factor: str, value: float) -> str:
    if factor == "lambda":
        return f"{value:g}"
    return str(int(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(lambda: {"tacc": [], "dar": []})
    with args.summary_csv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            factor, value = factor_info(row["factor"])
            key = (row["attack"], factor, value)
            grouped[key]["tacc"].append(float(row["tacc_mean10"]))
            grouped[key]["dar"].append(float(row["dar_mean10"]))

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    colors = {"tacc": "#1479b8", "dar": "#e26a2c"}
    markers = {"tacc": "o", "dar": "^"}
    for factor, (label, expected_values) in FACTORS.items():
        fig, axes = plt.subplots(2, len(ATTACKS), figsize=(16.2, 6.7), sharey="row")
        positions = list(range(len(expected_values)))
        for column, attack in enumerate(ATTACKS):
            for row_index, (metric, metric_label) in enumerate((("tacc", "TAcc"), ("dar", "DAR"))):
                axis = axes[row_index, column]
                points = []
                counts = []
                for position, value in enumerate(expected_values):
                    values = grouped.get((attack, factor, value), {}).get(metric, [])
                    if values:
                        points.append((position, statistics.fmean(values)))
                        counts.append(len(values))
                if points:
                    axis.plot(
                        [x for x, _ in points], [y for _, y in points],
                        color=colors[metric], marker=markers[metric], linewidth=2.0,
                        markersize=5.5, label=metric_label,
                    )
                observed = len(counts)
                max_n = max(counts, default=0)
                axis.set_title(f"({chr(97 + column)}) {ATTACK_LABELS[attack]}\n{observed}/{len(expected_values)} values, max n={max_n}" if row_index == 0 else "")
                axis.set_xticks(positions)
                axis.set_xticklabels([value_label(factor, value) for value in expected_values])
                axis.set_ylim(0.0, 1.05)
                axis.grid(True, linestyle=":", linewidth=1.2, color="#707070", alpha=0.85)
                for spine in axis.spines.values():
                    spine.set_color("#202020")
                    spine.set_linewidth(1.0)
                if row_index == 1:
                    axis.set_xlabel(label)
                if column == 0:
                    axis.set_ylabel("TAcc" if row_index == 0 else "DAR")
        fig.suptitle(f"Fashion-MNIST AE-SVDD sensitivity: {label}", y=1.01, fontsize=15)
        fig.tight_layout()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_dir / f"fashion-gnsfbd-{factor}.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
