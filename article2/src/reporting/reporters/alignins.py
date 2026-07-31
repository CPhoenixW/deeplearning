from .generic import GenericReporter


class AlignInsReporter(GenericReporter):
    title = "AlignIns"
    metric_key = "anomaly_score"
    metric_label = "Anomaly Score"
    participant_columns = (
        ("anomaly_score", "Anomaly Score"),
        ("tda", "TDA"),
        ("mpsa", "MPSA"),
    )
