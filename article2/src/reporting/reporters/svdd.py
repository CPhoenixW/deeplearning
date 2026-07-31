from __future__ import annotations

from typing import Any, Dict, Tuple

from .base import RoundReporter, _number


class SVDDReporter(RoundReporter):
    title = "AE-SVDD"
    server_order = (
        "svdd_loss",
        "recon_loss",
        "total_loss",
        "tau",
        "threshold",
        "center_norm",
        "z_variance",
        "kept",
        "num_clients",
    )

    def participant_metric_definitions(
        self, event: Dict[str, Any]
    ) -> Tuple[Tuple[str, str], ...]:
        if str(event["phase"]).lower() == "warmup":
            return (("reconstruction_loss", "Recon Loss"),)
        return (
            ("svdd_loss", "SVDD Loss"),
            ("reconstruction_loss", "Recon Loss"),
            ("total_loss", "Total Loss"),
        )

    def metric_definition(self, event: Dict[str, Any]) -> Tuple[str, str]:
        if str(event["phase"]).lower() == "warmup":
            return "reconstruction_loss", "Recon Loss"
        return "svdd_loss", "SVDD Loss"

    def server_metrics(self, event: Dict[str, Any]) -> Dict[str, Any]:
        payload = super().server_metrics(event)
        stats = event["stats"]
        payload["svdd_loss"] = _number(stats.svdd_loss)
        payload["recon_loss"] = _number(stats.recon_loss)
        payload["total_loss"] = _number(stats.total_loss)
        return payload


__all__ = ["SVDDReporter"]
