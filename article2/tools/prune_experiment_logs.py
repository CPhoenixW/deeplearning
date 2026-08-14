#!/usr/bin/env python3
"""Report and remove redundant federated-learning log artifacts.

Final ``*.json`` result files are the source of truth: they contain the run
metadata and all recorded rounds.  Timestamped ``*.jsonl`` files duplicate a
completed result, while console captures and PID files are operational
artifacts.  JSONL files without a matching final JSON are retained because
they may be the only record of an interrupted run.

The default mode is a dry run.  Pass ``--apply`` only after reviewing the
report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


DISPOSABLE_SUFFIXES = {"console.log", "recovery_console.log", "recovery_gpu.log"}


def _canonical_result_for_jsonl(path: Path) -> Path:
    """Return the non-timestamped JSON sibling for a JSONL result."""

    stem = path.name[: -len(".jsonl")]
    if "__" not in stem:
        return path.with_suffix(".json")
    base, _timestamp = stem.rsplit("__", 1)
    return path.with_name(base + ".json")


def _classify(path: Path) -> str:
    if path.name in DISPOSABLE_SUFFIXES or path.name.endswith(".pid"):
        return "operational"
    if ".ipynb_checkpoints" in path.parts:
        return "checkpoint"
    if path.suffix == ".jsonl":
        return "duplicate_jsonl" if _canonical_result_for_jsonl(path).exists() else "partial_jsonl"
    if path.suffix == ".json":
        return "final_json"
    if path.suffix == ".log":
        return "summary_log"
    return "other"


def collect(root: Path) -> tuple[dict[str, list[Path]], dict[str, int]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    sizes: Counter[str] = Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        category = _classify(path)
        groups[category].append(path)
        sizes[category] += path.stat().st_size
    return groups, sizes


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("log/stage_b"))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete redundant artifacts; without this flag only report them",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"log root does not exist: {root}")

    groups, sizes = collect(root)
    removable_categories = {"duplicate_jsonl", "operational", "checkpoint"}
    removable = [p for c in removable_categories for p in groups.get(c, [])]
    removable_size = sum(p.stat().st_size for p in removable)

    print(f"root: {root}")
    print("category                  files       size")
    print("-" * 52)
    for category in sorted(groups):
        print(f"{category:24s} {len(groups[category]):5d} {_format_bytes(sizes[category]):>12s}")
    print(f"removable (planned)       {len(removable):5d} {_format_bytes(removable_size):>12s}")
    print("partial_jsonl files are retained because they have no final JSON sibling.")

    if not args.apply:
        print("dry-run only; pass --apply to remove the planned artifacts")
        return 0

    deleted = 0
    for path in removable:
        try:
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
    print(f"deleted: {deleted} files ({_format_bytes(removable_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
