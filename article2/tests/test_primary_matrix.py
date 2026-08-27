from __future__ import annotations

import json
from pathlib import Path

from tools.generate_primary_matrix import (
    AG_NEWS_ATTACKS,
    DEFENSES,
    IMAGE_ATTACKS,
    SEEDS,
    write_matrix,
)
from tools.run_stage0_smoke import CASES, DEFENSES as STAGE0_DEFENSES, build_jobs


def test_primary_matrix_has_816_jobs_and_ag_news_restriction(tmp_path) -> None:
    jobs = write_matrix(tmp_path / "primary", rounds=12)

    assert len(jobs) == 816
    assert len({job.config_path for job in jobs}) == 816
    assert {job.seed for job in jobs} == set(SEEDS)
    assert {job.defense for job in jobs} == set(DEFENSES)
    assert {job.attack for job in jobs if job.task == "ag_news"} == set(AG_NEWS_ATTACKS)
    assert not any(job.task == "ag_news" and job.attack in {"bd", "mix"} for job in jobs)
    assert {job.attack for job in jobs if job.task == "mnist"} == set(IMAGE_ATTACKS)

    sample = next(job for job in jobs if job.task == "cifar10" and job.attack == "lie")
    payload = json.loads(Path(sample.config_path).read_text(encoding="utf-8"))
    overrides = payload["fed_config_overrides"]
    assert overrides["svdd_lambda"] == 0.5
    assert "phase1_score_mode" not in overrides
    assert "phase2_score_mode" not in overrides
    assert "svdd_input_mode" not in overrides
    assert "lie_z_override" not in overrides
    assert overrides["mixed_attack_types"] == "lf,bd,gn,sf,lie,minmax,minsum"
    assert overrides["total_rounds"] == 12


def test_stage0_matrix_covers_four_tasks_and_primary_defenses(tmp_path) -> None:
    jobs = build_jobs(tmp_path / "stage0", rounds=12)

    assert len(jobs) == len(CASES) * len(STAGE0_DEFENSES) == 32
    assert {task for task, _attack, _defense, _config, _output in jobs} == {
        "mnist",
        "fashion_mnist",
        "cifar10",
        "ag_news",
    }
    assert not any(
        task == "ag_news" and attack in {"bd", "mix"}
        for task, attack, _defense, _config, _output in jobs
    )
