from __future__ import annotations

import argparse

from tools.supervise_rq3_mechanism_matrix import _parse_workers, _runner_command


def test_supervisor_retries_with_decreasing_concurrency() -> None:
    assert _parse_workers("12,8,4,2,1") == (12, 8, 4, 2, 1)


def test_supervisor_builds_the_confirmed_rq3_command() -> None:
    args = argparse.Namespace(
        python_bin="python",
        task="fashion_mnist",
        mechanisms="fedavg,p1_only,p2_only,full",
        attacks="gn,lie,bd,mix",
        malicious_ratios="0.1,0.2,0.3,0.4",
        seeds="42",
        rounds=100,
        gpus="0,2",
        output_root="log/rq3",
    )
    command = _runner_command(args, 12)
    assert command[0:2] == ["python", "tools/run_rq3_mechanism_matrix.py"]
    assert command[command.index("--gpus") + 1] == "0,2"
    assert command[command.index("--workers-per-gpu") + 1] == "12"
    assert command[command.index("--rounds") + 1] == "100"
