from .generic import GenericReporter


class FLANDERSReporter(GenericReporter):
    title = "FLANDERS"
    metric_key = "mar_score"
    metric_label = "MAR Score"
    participant_columns = (("mar_score", "MAR Score"),)
