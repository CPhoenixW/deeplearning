#!/usr/bin/env python3
"""Plot completed Fashion-MNIST AE-SVDD attack runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ATTACKS = ("gn", "sf", "lf", "bd")
LABELS = {"gn": "GN", "sf": "SF", "lf": "LF", "bd": "BD"}
SEEDS = (42, 43, 44)


def _load(root: Path, attack: str, seed: int) -> dict:
    path = root / attack / f"seed_{seed}" / f"fashion_mnist__{attack}__svdd.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    jsonl_paths = sorted(path.parent.glob(f"{path.stem}__*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(path)
    rounds = []
    for line in jsonl_paths[-1].read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("record_type") == "round":
            rounds.append(record)
    if not rounds:
        raise ValueError(f"No round records found in {jsonl_paths[-1]}")
    return {"rounds": rounds, "partial": True}


def _series(payload: dict, key: str) -> list[float]:
    return [float(item["evaluation"][key]) for item in payload["rounds"]]


def _mean_std(values: list[list[float]]) -> tuple[list[float], list[float]]:
    length = min(len(item) for item in values)
    means, stds = [], []
    for index in range(length):
        column = [item[index] for item in values]
        means.append(statistics.fmean(column))
        stds.append(statistics.stdev(column) if len(column) > 1 else 0.0)
    return means, stds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        attack: {seed: _load(args.root, attack, seed) for seed in SEEDS}
        for attack in ATTACKS
    }
    partial = [
        f"{LABELS[attack]} seed{seed} ({len(runs[attack][seed]['rounds'])} rounds)"
        for attack in ATTACKS
        for seed in SEEDS
        if runs[attack][seed].get("partial")
    ]

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    colors = {"dar": "#d95f02", "backdoor_asr": "#1b9e77"}

    # Main per-round curves: DAR for every attack and ASR for BD.
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharex=True)
    for axis, metric, ylabel in (
        (axes[0], "dar", "DAR"),
        (axes[1], "backdoor_asr", "Backdoor ASR"),
    ):
        attacks = ("bd",) if metric == "backdoor_asr" else ATTACKS
        for attack in attacks:
            values = [_series(runs[attack][seed], metric) for seed in SEEDS]
            mean, std = _mean_std(values)
            rounds = list(range(1, len(mean) + 1))
            color = colors[metric] if metric == "backdoor_asr" else None
            if color is None:
                color = {"gn": "#377eb8", "sf": "#4daf4a", "lf": "#984ea3", "bd": "#d95f02"}[attack]
            axis.plot(rounds, mean, color=color, linewidth=2.0, label=LABELS[attack])
            axis.fill_between(
                rounds,
                [max(0.0, x - y) for x, y in zip(mean, std)],
                [min(1.0, x + y) for x, y in zip(mean, std)],
                color=color,
                alpha=0.14,
            )
        axis.set_xlabel("Communication round")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, linestyle=":", alpha=0.7)
        axis.legend(frameon=True)
    suffix = " [partial: " + ", ".join(partial) + "]" if partial else ""
    fig.suptitle("Fashion-MNIST AE-SVDD: absolute-parameter input" + suffix, fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fashion-svdd-absolute-curves.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Final ten-round summary across seeds.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    attacks = list(ATTACKS)
    positions = list(range(len(attacks)))
    for axis, metric, ylabel in (
        (axes[0], "dar", "DAR"),
        (axes[1], "backdoor_asr", "Backdoor ASR"),
    ):
        plotted_attacks = ("bd",) if metric == "backdoor_asr" else tuple(ATTACKS)
        plotted = []
        for attack in plotted_attacks:
            seed_means = [
                statistics.fmean(_series(runs[attack][seed], metric)[-10:])
                for seed in SEEDS
            ]
            if seed_means:
                plotted.append(
                    (
                        attack,
                        statistics.fmean(seed_means),
                        statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
                    )
                )
        x = [positions[attacks.index(attack)] for attack, _, _ in plotted]
        y = [mean for _, mean, _ in plotted]
        error = [std for _, _, std in plotted]
        axis.errorbar(x, y, yerr=error, fmt="o", color="#2c3e50", capsize=5, linewidth=1.8, markersize=7)
        axis.set_xticks(positions if metric == "dar" else [positions[attacks.index("bd")]])
        axis.set_xticklabels(
            [LABELS[attack] for attack in attacks]
            if metric == "dar"
            else [LABELS["bd"]]
        )
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, axis="y", linestyle=":", alpha=0.7)
    fig.suptitle("Fashion-MNIST AE-SVDD: final-10-round summary" + suffix, fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fashion-svdd-absolute-summary.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
