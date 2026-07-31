from .generic import GenericReporter


class TrimmedMeanReporter(GenericReporter):
    title = "Trimmed Mean"
    metric_key = "aggregation_weight"
    metric_label = "Aggregation Weight"
    participant_columns = (("aggregation_weight", "Agg Weight"),)


__all__ = ["TrimmedMeanReporter"]
