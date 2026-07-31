from .generic import GenericReporter


class FLGMMReporter(GenericReporter):
    title = "FLGMM"
    metric_key = "standardized_distance"
    metric_label = "Standardized Distance"
    participant_columns = (("standardized_distance", "Std Distance"),)
