from .generic import GenericReporter


class LASAReporter(GenericReporter):
    title = "LASA"
    metric_key = "benign_layer_ratio"
    metric_label = "Benign Layer Ratio"
    participant_columns = (("benign_layer_ratio", "Benign Layer Ratio"),)
