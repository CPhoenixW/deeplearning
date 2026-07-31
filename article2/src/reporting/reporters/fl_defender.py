from .generic import GenericReporter


class FLDefenderReporter(GenericReporter):
    title = "FL-Defender"
    metric_key = "trust_score"
    metric_label = "Trust Score"
    participant_columns = (("trust_score", "Trust Score"),)
