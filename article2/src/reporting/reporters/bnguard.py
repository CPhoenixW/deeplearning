from .generic import GenericReporter


class BNGuardReporter(GenericReporter):
    title = "BNGuard"
    metric_key = "bn_distance"
    metric_label = "BN Distance"
    participant_columns = (("bn_distance", "BN Distance"),)
