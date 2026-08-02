"""Tests for the JSON-pipeline result analyzer."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analyze_agnews_logs import parse_one_log


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
