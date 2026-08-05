#!/usr/bin/env python3
"""Wait for sweep wave 1 to finish, then launch the remaining variants.

Completion is based on the final JSON contract (all attacks, exact round count,
and exact effective hyperparameters), rather than transient process IDs.  The
script is safe to restart: the underlying runner skips only valid results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from run_sf_weight_tau_sweep import DEFAULT_ATTACKS, _complete


FIRST_WAVE = ("recon2_tau32", "recon4_tau32", "recon2_tau21")
SECOND_WAVE = ("recon4_tau21", "recon2_tau31", "recon4_tau31")


def _parse_csv(value: str, *, cast: type = str) -> tuple[object, ...]:
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("CSV argument must contain at least one value")
    return values


def _wave_status(
    root: Path, variants: tuple[str, ...], seed: int, attacks: tuple[str, ...], rounds: int
) -> dict[str, bool]:
    return {
        variant: _complete(root / variant / f"seed_{seed}", attacks, rounds)
        for variant in variants
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("log/sf_weight_tau_sweep"),
    )
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--attack-workers", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--python", dest="python_bin", default=".venv/bin/python")
    args = parser.parse_args()

    gpus = _parse_csv(args.gpus, cast=int)
    attacks = _parse_csv(args.attacks, cast=str)
    if len(gpus) != 3:
        parser.error("--gpus must contain exactly three GPU ids")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")

    root = args.output_root.resolve()
    project_root = Path(__file__).resolve().parents[1]
    runner = Path(__file__).with_name("run_sf_weight_tau_sweep.py")
    last_status: dict[str, bool] | None = None

    while True:
        status = _wave_status(
            root,
            FIRST_WAVE,
            int(args.seed),
            tuple(str(item) for item in attacks),
            int(args.rounds),
        )
        if status != last_status:
            print(f"wave1 status={status}", flush=True)
            last_status = status
        if all(status.values()):
            break
        time.sleep(int(args.poll_seconds))

    print(f"wave1 complete; launching wave2={SECOND_WAVE}", flush=True)
    children: list[tuple[str, subprocess.Popen[bytes]]] = []
    for variant, gpu in zip(SECOND_WAVE, gpus, strict=True):
        command = [
            sys.executable,
            str(runner),
            "--variant",
            variant,
            "--gpus",
            str(gpu),
            "--seeds",
            str(args.seed),
            "--attacks",
            ",".join(str(item) for item in attacks),
            "--rounds",
            str(args.rounds),
            "--attack-workers",
            str(args.attack_workers),
            "--output-root",
            str(root),
            "--python",
            str(args.python_bin),
        ]
        child = subprocess.Popen(command, cwd=project_root, start_new_session=True)
        children.append((variant, child))
        print(f"started variant={variant} gpu={gpu} pid={child.pid}", flush=True)

    result = 0
    for variant, child in children:
        code = child.wait()
        print(f"finished variant={variant} returncode={code}", flush=True)
        if code != 0 and result == 0:
            result = int(code)
    if result == 0:
        status = _wave_status(
            root,
            SECOND_WAVE,
            int(args.seed),
            tuple(str(item) for item in attacks),
            int(args.rounds),
        )
        print(f"wave2 status={status}", flush=True)
        if not all(status.values()):
            result = 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
