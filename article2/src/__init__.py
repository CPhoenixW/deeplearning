"""
Federated learning with AE-SVDD defenses (CIFAR-10, Fashion-MNIST, AG News).

Layout (project root = parent of ``src/``)::

    article2/
      src/          # Python package (this directory)
      data/         # datasets (CIFAR-10, Fashion-MNIST, AG News cache, …)
      log/          # experiment JSON outputs from ``run_matrix`` / ``run_silent``

Default ``data_root`` / ``run_matrix --log-dir`` are resolved from the project
root (parent of ``src/``), not from the process working directory.

Typical invocation::

    python -m src.run_matrix --list
    python -m src.run_matrix --task ag_news --defenses svdd --rounds 50
"""

__all__: list[str] = []
