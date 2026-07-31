from .generic import GenericReporter


class FedAvgReporter(GenericReporter):
    title = "FedAvg"
    metric_key = "aggregation_weight"
    metric_label = "Aggregation Weight"
    participant_columns = (("aggregation_weight", "Agg Weight"),)


__all__ = ["FedAvgReporter"]
