from __future__ import annotations

import json
from pathlib import Path

from tools.run_rq3_mechanism_matrix import (
    MECHANISMS,
    build_jobs,
    iter_specs,
)


def test_rq3_matrix_has_explicit_mechanism_endpoints(tmp_path: Path) -> None:
    specs = list(
        iter_specs(
            mechanisms=MECHANISMS,
            attacks=("gn", "lie", "bd", "mix"),
            ratios=(0.1, 0.2, 0.3, 0.4),
            seeds=(42,),
        )
    )
    assert len(specs) == 68
    assert sum(item[1] == "none" and item[2] == 0.0 for item in specs) == 4
    assert {item[0] for item in specs} == set(MECHANISMS)

    jobs = build_jobs(
        tmp_path / "rq3",
        task="fashion_mnist",
        mechanisms=MECHANISMS,
        attacks=("gn", "lie", "bd", "mix"),
        ratios=(0.1, 0.2, 0.3, 0.4),
        seeds=(42,),
        rounds=100,
    )
    assert len(jobs) == 68
    payloads = [json.loads(path.read_text(encoding="utf-8")) for *_rest, path, _out in jobs]
    by_mechanism = {
        path.parts[-4]: json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "rq3").rglob("seed_42.json")
    }
    assert by_mechanism["p1_only"]["fed_config_overrides"]["phase1_rounds"] == 100
    assert by_mechanism["p2_only"]["fed_config_overrides"]["phase1_rounds"] == 0
    assert by_mechanism["full"]["fed_config_overrides"]["phase1_rounds"] == 15
    assert by_mechanism["full"]["fed_config_overrides"]["phase2_score_mode"] == "svdd"
    assert by_mechanism["full"]["fed_config_overrides"]["alpha"] == 0.5
    assert "lie_z_override" not in by_mechanism["full"]["fed_config_overrides"]
    assert all(payload["fed_config_overrides"]["num_malicious"] == 0 for payload in payloads if payload["attacks"] == "none")


def test_rq3_result_paths_are_resumable(tmp_path: Path) -> None:
    kwargs = dict(
        task="fashion_mnist",
        mechanisms=MECHANISMS,
        attacks=("gn",),
        ratios=(0.1,),
        seeds=(42,),
        rounds=20,
    )
    first = build_jobs(tmp_path / "rq3", **kwargs)
    assert len(first) == 8
    for *_prefix, config_path, output_dir in first[:2]:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        effective = payload["fed_config_overrides"]
        result = {
            "meta": {
                "task": "fashion_mnist",
                "attack": payload["attacks"],
                "defense": payload["defenses"],
                "total_rounds": 20,
                "effective_config": {**effective, "num_clients": 100, "num_benign": 100},
            },
            "rounds": [{}] * 20,
        }
        (output_dir / f"fashion_mnist__{payload['attacks']}__{payload['defenses']}.json").write_text(json.dumps(result), encoding="utf-8")
    second = build_jobs(tmp_path / "rq3", **kwargs)
    assert len(second) == 7
