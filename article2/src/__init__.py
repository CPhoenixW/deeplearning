"""JSON-driven federated malicious-client detection experiments.

The only experiment execution path is ``python -m src.pipeline --config ...``.
Client attack implementations, including mixed attack composition, live
exclusively in :mod:`src.attacks`.
Defense implementations live exclusively in :mod:`src.defenses`.
"""

__all__: list[str] = []
