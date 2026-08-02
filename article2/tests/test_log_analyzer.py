"""Tests for the JSON-pipeline result analyzer."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analyze_agnews_logs import parse_one_log
from src.config import FedConfig
from src.pipeline_core.contracts import PipelineContext
from src.pipeline_core.result import StructuredResultWriter


def test_log_analyzer_reads_pipeline_result_schema() -> None:
    payload = {
        "meta": {"task": "ag_news", "attack": "mix", "defense": "svdd"},
        "rounds": [
            {
                "evaluation": {
                    "accuracy": 0.25,
                    "dar": 0.75,
                    "dpr": 0.5,
                }
            },
            {
                "evaluation": {
                    "accuracy": 0.5,
                    "dar": 1.0,
                    "dpr": 0.75,
                }
            },
        ],
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "ag_news__mix__svdd.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        parsed = parse_one_log(path, last_n=2)

    assert parsed is not None
    assert parsed["task_name"] == "ag_news"
    assert parsed["avg_acc"] == 0.375
    assert parsed["avg_dar"] == 0.875
    assert parsed["avg_dpr"] == 0.625


def test_result_metadata_records_effective_training_config() -> None:
    config = FedConfig(
        num_clients=2,
        num_benign=2,
        total_rounds=0,
        client_lr=0.03,
        client_weight_decay=0.0001,
        attack_type="none",
        defense_type="avg",
        aggregation_method="avg",
    )
    with TemporaryDirectory() as directory:
        context = PipelineContext(
            config=config,
            task_name="mnist",
            attack_name="none",
            defense_name="avg",
            output_dir=Path(directory),
            config_files={},
        )
        result_path = StructuredResultWriter(context).write()
        payload = json.loads(result_path.read_text(encoding="utf-8"))

    effective = payload["meta"]["effective_config"]
    assert effective["client_lr"] == 0.03
    assert effective["client_weight_decay"] == 0.0001
    assert effective["seed"] == config.seed
