"""Tests for Stage-A JSON generation, ranking and promotion."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.stage_a_calibration import (
    Trial,
    _is_complete,
    _pipeline_payload,
    _worker_gpu_ids,
    build_trials,
    load_manifest,
    promote_manifest,
    select_candidates,
)


def test_stage_a_expands_multiple_workers_per_gpu() -> None:
    manifest = {
        "gpus": [0, 1, 2],
        "workers_per_gpu": 3,
        "workers_per_gpu_by_task": {"cifar10": 2},
    }
    assert _worker_gpu_ids(manifest, None, None) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert _worker_gpu_ids(manifest, None, None, task="cifar10") == [0, 0, 1, 1, 2, 2]
    assert _worker_gpu_ids(manifest, "2,0", 2) == [2, 2, 0, 0]


def test_stage_a_screen_manifest_generates_expected_grid() -> None:
    _path, manifest = load_manifest("configs/stage_a_screen.json")
    trials = build_trials(manifest)
    assert len(trials) == 80
    assert {
        task: sum(1 for trial in trials if trial.task == task)
        for task in manifest["tasks"]
    } == {
        "mnist": 20,
        "fashion_mnist": 20,
        "cifar10": 20,
        "ag_news": 20,
    }

    mnist_trial = next(trial for trial in trials if trial.task == "mnist")
    payload = _pipeline_payload(manifest, mnist_trial)
    assert payload["attacks"] == "none"
    assert payload["defenses"] == "avg"
    assert payload["fed_config_overrides"]["num_malicious"] == 0
    assert payload["fed_config_overrides"]["hf_datasets_offline"] is True
    assert payload["fed_config_overrides"]["batch_size"] == 64
    assert payload["fed_config_overrides"]["client_batch_group_size"] == 25
    cifar_trial = next(trial for trial in trials if trial.task == "cifar10")
    cifar_payload = _pipeline_payload(manifest, cifar_trial)
    assert cifar_payload["fed_config_overrides"]["client_batch_group_size"] == 2
    ag_trial = next(trial for trial in trials if trial.task == "ag_news")
    ag_payload = _pipeline_payload(manifest, ag_trial)
    assert ag_payload["fed_config_overrides"]["client_batch_group_size"] == 5


def test_covid19_stage_a_manifest_generates_isolated_grid() -> None:
    _path, manifest = load_manifest("configs/stage_a_covid19_screen.json")
    trials = build_trials(manifest)
    assert len(trials) == 16
    assert {trial.task for trial in trials} == {"covid19"}
    trial = trials[0]
    payload = _pipeline_payload(manifest, trial)
    assert payload["task"] == "covid19"
    assert payload["attacks"] == "none"
    assert payload["defenses"] == "avg"
    assert payload["fed_config_overrides"]["num_malicious"] == 0
    assert payload["fed_config_overrides"]["client_batch_group_size"] == 10


def _write_result(trial: Trial, accuracies: list[float]) -> None:
    trial.config_path.write_text(
        json.dumps(
            {
                "fed_config_overrides": {
                    "client_lr": trial.client_lr,
                    "client_weight_decay": trial.client_weight_decay,
                    "seed": trial.seed,
                    "batch_size": 64,
                }
            }
        ),
        encoding="utf-8",
    )
    trial.output_dir.mkdir(parents=True, exist_ok=True)
    trial.result_path.write_text(
        json.dumps(
            {
                "meta": {
                    "task": trial.task,
                    "attack": "none",
                    "defense": "avg",
                    "total_rounds": len(accuracies),
                    "effective_config": {
                        "client_lr": trial.client_lr,
                        "client_weight_decay": trial.client_weight_decay,
                        "seed": trial.seed,
                        "batch_size": 64,
                    },
                },
                "rounds": [
                    {"evaluation": {"accuracy": accuracy}}
                    for accuracy in accuracies
                ],
            }
        ),
        encoding="utf-8",
    )


def test_stage_a_completion_rejects_changed_effective_config() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        trial = Trial(
            task="mnist",
            client_lr=0.05,
            client_weight_decay=0.0,
            seed=42,
            config_path=root / "trial.json",
            output_dir=root / "result",
        )
        trial.config_path.write_text(
            json.dumps(
                {
                    "fed_config_overrides": {
                        "client_lr": 0.05,
                        "client_weight_decay": 0.0,
                        "seed": 42,
                        "batch_size": 64,
                    }
                }
            ),
            encoding="utf-8",
        )
        _write_result(trial, [0.5, 0.6])
        assert _is_complete(trial, 2)

        payload = json.loads(trial.config_path.read_text(encoding="utf-8"))
        payload["fed_config_overrides"]["batch_size"] = 256
        trial.config_path.write_text(json.dumps(payload), encoding="utf-8")
        assert not _is_complete(trial, 2)


def test_stage_a_selection_uses_only_clean_tacc_and_promotes_top_candidate() -> None:
    manifest = {
        "name": "test_screen",
        "tasks": ["mnist"],
        "client_lrs": [0.01, 0.1],
        "client_weight_decays": [0.0],
        "seeds": [42, 43],
        "score_last_n_rounds": 2,
        "common_overrides": {"total_rounds": 2},
        "promotion_defaults": {},
    }
    with TemporaryDirectory() as directory:
        root = Path(directory)
        trials = [
            Trial(
                task="mnist",
                client_lr=learning_rate,
                client_weight_decay=0.0,
                seed=seed,
                config_path=root / f"lr_{learning_rate}_seed_{seed}.json",
                output_dir=root / f"lr_{learning_rate}" / f"seed_{seed}",
            )
            for learning_rate in (0.01, 0.1)
            for seed in (42, 43)
        ]
        for trial in trials:
            score = 0.9 if trial.client_lr == 0.01 else 0.7
            _write_result(trial, [score - 0.1, score])

        selection = select_candidates(manifest, trials)
        assert selection["selection_uses_only"] == "clean_tacc"
        assert selection["selected_by_task"]["mnist"] == {
            "client_lr": 0.01,
            "client_weight_decay": 0.0,
        }
        promoted = promote_manifest(
            manifest,
            selection,
            top_k=1,
            rounds=300,
            seeds=[42, 43, 44],
        )

    assert promoted["task_candidates"]["mnist"] == [
        {"client_lr": 0.01, "client_weight_decay": 0.0}
    ]
    assert promoted["seeds"] == [42, 43, 44]
    assert promoted["common_overrides"]["total_rounds"] == 300
