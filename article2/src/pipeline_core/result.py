from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ..reporting.reporters import get_reporter
from ..clients import mixed_attack_for_client
from .contracts import PipelineContext


def _json_safe(value: Any) -> Any:
    """Convert nested runtime values to strict JSON values."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class StructuredResultWriter:
    def __init__(self, context: PipelineContext) -> None:
        self.context = context
        self.reporter = get_reporter(context.defense_name)
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.run_id, self._jsonl_path = self._create_jsonl_stream()

    @property
    def jsonl_path(self) -> Path:
        """Path of the incrementally written round-level JSONL stream."""

        return self._jsonl_path

    def _create_jsonl_stream(self) -> tuple[str, Path]:
        cfg = self.context.config
        self.context.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        stem = (
            f"{self.context.task_name}__{self.context.attack_name}__"
            f"{self.context.defense_name}__"
        )
        meta = {
            "record_type": "meta",
            "run_id": timestamp,
            "task": self.context.task_name,
            "attack": self.context.attack_name,
            "defense": self.context.defense_name,
            "started_at": self.started_at,
            "num_clients": cfg.num_clients,
            "num_benign": cfg.num_benign,
            "num_malicious": cfg.num_clients - cfg.num_benign,
            "planned_rounds": cfg.total_rounds,
            "config_files": dict(self.context.config_files),
            "applied_hyperparameters": dict(self.context.applied_hyperparameters),
            "mixed_attack_types": getattr(cfg, "mixed_attack_types", "lf,bd,gn"),
            "mixed_attack_assignments": (
                {
                    str(cid): mixed_attack_for_client(cfg, cid)
                    for cid in range(cfg.num_benign, cfg.num_clients)
                }
                if self.context.attack_name == "mix" else {}
            ),
        }
        for attempt in range(100):
            run_id = timestamp if attempt == 0 else f"{timestamp}-{attempt}"
            path = self.context.output_dir / f"{stem}{run_id}.jsonl"
            try:
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            _json_safe({**meta, "run_id": run_id}),
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                return run_id, path
            except FileExistsError:
                continue
        raise RuntimeError("Unable to allocate a unique JSONL run file.")

    def observe(self, event: Dict[str, Any]) -> None:
        round_payload = self.reporter.build_round(event)
        self.context.rounds.append(round_payload)
        record = {"record_type": "round", **round_payload}
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    _json_safe(record),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def write(self) -> Path:
        cfg = self.context.config
        payload = {
            "meta": {
                "task": self.context.task_name,
                "attack": self.context.attack_name,
                "defense": self.context.defense_name,
                "started_at": self.started_at,
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "num_clients": cfg.num_clients,
                "num_benign": cfg.num_benign,
                "num_malicious": cfg.num_clients - cfg.num_benign,
                "total_rounds": cfg.total_rounds,
                "config_files": dict(self.context.config_files),
                "applied_hyperparameters": dict(self.context.applied_hyperparameters),
                "mixed_attack_types": getattr(cfg, "mixed_attack_types", "lf,bd,gn"),
                "mixed_attack_assignments": (
                    {
                        str(cid): mixed_attack_for_client(cfg, cid)
                        for cid in range(cfg.num_benign, cfg.num_clients)
                    }
                    if self.context.attack_name == "mix" else {}
                ),
            },
            "rounds": self.context.rounds,
        }
        self.context.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.context.output_dir / (
            f"{self.context.task_name}__{self.context.attack_name}__"
            f"{self.context.defense_name}.json"
        )
        path.write_text(
            json.dumps(
                _json_safe(payload),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.context.result_path = path
        return path
