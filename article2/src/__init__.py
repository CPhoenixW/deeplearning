"""
Federated learning with AE-SVDD defenses (CIFAR-10, Fashion-MNIST, AG News).

Layout (project root = parent of ``src/``)::

    article2/
      src/          # Python package (this directory)
      data/         # datasets (CIFAR-10, Fashion-MNIST, AG News cache, …)
      log/          # experiment JSON outputs from ``run_matrix`` / ``run_silent``

Run from project root::

    python -m src.run_matrix --list
    python -m src.run_matrix --task ag_news --defenses svdd --rounds 50
"""

__all__: list[str] = []
