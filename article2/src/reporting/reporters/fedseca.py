from .generic import GenericReporter


class FedSECAReporter(GenericReporter):
    title = "FedSECA"
    metric_key = "concordance"
    metric_label = "Concordance"
    participant_columns = (
        ("concordance", "Concordance"),
        ("cosine_similarity", "Cosine Sim"),
    )
