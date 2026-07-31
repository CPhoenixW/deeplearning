from .generic import GenericReporter


class MultiKrumReporter(GenericReporter):
    title = "Multi-Krum"
    metric_key = "krum_score"
    metric_label = "Krum Score"
    participant_columns = (("krum_score", "Krum Score"),)
