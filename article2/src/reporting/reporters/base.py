from __future__ import annotations

import math
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Tuple

import torch

from ..contracts import ParticipantResult


def _number(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return value
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _key(label: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return value or "metric"


class RoundReporter:
    """Defense-owned JSON and console projection of a common round result."""

    title = "Federated Aggregation"
    metric_key = "score"
    metric_label = "Score"
    server_order: Tuple[str, ...] = ("kept", "num_clients")
    # Subclasses own the participant columns shown in the console. The first
    # column is also the fallback for the legacy ``d`` score vector.
    participant_columns: Tuple[Tuple[str, str], ...] = ()

    def metric_definition(self, event: Dict[str, Any]) -> Tuple[str, str]:
        return self.metric_key, self.metric_label

    def participant_metric_definitions(
        self, event: Dict[str, Any]
    ) -> Tuple[Tuple[str, str], ...]:
        """Return the defense-specific participant columns for this phase."""

        return self.participant_columns or (self.metric_definition(event),)

    def server_metrics(self, event: Dict[str, Any]) -> Dict[str, Any]:
        stats = event["stats"]
        server = {}
        for key, value in stats.server_metrics.items():
            parsed = _number(value)
            if parsed is not None:
                server[key] = parsed
        for label, value in stats.monitor_items:
            server.setdefault(_key(label), _number(value))
        for key, value in stats.diagnostics.items():
            server[f"diagnostic_{key}"] = _number(value)
        return server

    def participant_metrics(
        self,
        event: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any]:
        stats = event["stats"]
        metric_key, _label = self.metric_definition(event)
        columns = self.participant_metric_definitions(event)
        metrics: Dict[str, Any] = {}
        for key, _label in columns:
            values = stats.participant_metrics.get(key)
            if values is not None:
                value = values.reshape(-1)[index]
                metrics[key] = _number(value)
            elif key == metric_key:
                metrics[key] = _number(stats.participant_scores[index])
            else:
                metrics[key] = None

        # Preserve strategy-provided diagnostics in JSON even when a reporter
        # chooses not to put every diagnostic in the compact console table.
        for key, values in stats.participant_metrics.items():
            if key not in metrics:
                metrics[key] = _number(values.reshape(-1)[index])
        return metrics

    def build_round(self, event: Dict[str, Any]) -> Dict[str, Any]:
        stats = event["stats"]
        ground_truth = event["ground_truth"].detach().cpu().reshape(-1)
        scores = stats.participant_scores.detach().cpu().reshape(-1)
        weights = stats.participant_weights.detach().cpu().reshape(-1)
        accepted = stats.accepted_mask.detach().cpu().reshape(-1)
        if not (len(ground_truth) == len(scores) == len(weights) == len(accepted)):
            raise ValueError("Participant output vectors must have equal lengths")
        participants: List[Dict[str, Any]] = []
        for index in range(int(scores.numel())):
            participant = asdict(
                ParticipantResult(
                    client_id=index,
                    role="benign" if int(ground_truth[index].item()) == 1 else "malicious",
                    accepted=bool(accepted[index].item() >= 0.5),
                    weight=float(weights[index].item()),
                    metrics=self.participant_metrics(event, index),
                )
            )
            participant["id"] = participant.pop("client_id")
            participants.append(participant)
        evaluation = {
            "accuracy": float(event["test_acc"]),
            "correct": int(event["test_correct"]),
            "total": int(event["test_total"]),
            "tpr": float(event["tpr"]),
            "fpr": float(event["fpr"]),
            "dar": float(event["dar"]),
            "dpr": float(event["dpr"]),
            "rr": float(event["rr"]),
            "reject_rate": float(event["reject_rate"]),
        }
        if event.get("backdoor_asr") is not None:
            evaluation["backdoor_asr"] = float(event["backdoor_asr"])
        return {
            "round": int(event["round"]),
            "phase": str(event["phase"]),
            "server": self.server_metrics(event),
            "participants": participants,
            "evaluation": evaluation,
        }

    def ordered_server_metrics(
        self,
        server: Dict[str, Any],
    ) -> Iterable[Tuple[str, Any]]:
        seen = set()
        for key in self.server_order:
            if key in server:
                seen.add(key)
                yield key, server[key]
        for key in sorted(server):
            if key not in seen and not key.startswith("diagnostic_"):
                yield key, server[key]

    def print_console(self, event: Dict[str, Any]) -> None:
        payload = self.build_round(event)
        columns = self.participant_metric_definitions(event)
        print(f"\n=== Round {payload['round']} | {self.title} | {payload['phase']} ===")
        print("Server Metrics")
        for key, value in self.ordered_server_metrics(payload["server"]):
            print(f"  {key}: {value}")
        evaluation = payload["evaluation"]
        evaluation_line = (
            "Evaluation  "
            f"accuracy={evaluation['accuracy']:.4f}  "
            f"TPR={evaluation['tpr']:.4f}  FPR={evaluation['fpr']:.4f}"
        )
        if "backdoor_asr" in evaluation:
            evaluation_line += f"  ASR={evaluation['backdoor_asr']:.4f}"
        print(evaluation_line)
        print(f"Participant Metrics ({self.title})")
        headers = ["ID", "Role", "Accepted", "Weight"] + [label for _key, label in columns]
        rows = []
        for participant in payload["participants"]:
            values = [
                str(participant["id"]),
                str(participant["role"]),
                str(participant["accepted"]),
                f"{participant['weight']:.6f}",
            ]
            for key, _label in columns:
                value = participant["metrics"].get(key)
                values.append("-" if value is None else f"{float(value):.6f}")
            rows.append(values)

        widths = [len(header) for header in headers]
        for row in rows:
            for index, value in enumerate(row):
                widths[index] = max(widths[index], len(value))

        def format_row(values: List[str]) -> str:
            return "  " + "  ".join(
                value.rjust(widths[index]) if index in {0, 3} else value.ljust(widths[index])
                for index, value in enumerate(values)
            )

        print(format_row(headers))
        for row in rows:
            print(format_row(row))


__all__ = ["RoundReporter", "_number"]
