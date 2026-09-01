"""Regression tests for the FedDMC LIT attack port."""

from __future__ import annotations

import types

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.attacks import (
    FEDDMC_LIT_POISON_RATIO,
    FEDDMC_LIT_Z,
    LitAttack,
    attack_metadata,
    rewrite_lit_uploads,
)
from src.config import FedConfig


def _state(value: float) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([value], dtype=torch.float32),
        "counter": torch.tensor(0, dtype=torch.long),
    }


def test_lit_matches_released_feddmc_coordinate_clip() -> None:
    config = FedConfig(
        num_clients=4,
        num_benign=2,
        attack_type="lit",
        client_lr=1.0,
    )
    loader = DataLoader(
        TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)),
        batch_size=1,
    )
    model_fn = lambda: torch.nn.Linear(1, 1, bias=False)
    malicious = [
        LitAttack(client_id, torch.device("cpu"), config, loader, model_fn)
        for client_id in (2, 3)
    ]

    # Make the second (backdoor) training pass deterministic.  The LIT algebra
    # itself is what this test isolates; local backdoor SGD is covered by the
    # shared client/backdoor tests.
    for client in malicious:
        client.backdoor_step = types.MethodType(
            lambda self, start_state: _state(10.0), client
        )

    global_state = _state(0.0)
    # FedDMC computes the reference gradient statistics from the malicious
    # clients' ordinary pre-attack updates: g=(global-local)/lr -> {1, 3}.
    client_states = [_state(0.2), _state(-0.2), _state(-1.0), _state(-3.0)]
    clients = [None, None, *malicious]

    rewrite_lit_uploads(
        config,
        global_state,
        client_states,
        (2, 3),
        clients,
        parameter_names=("weight",),
    )

    mean = torch.tensor(2.0)
    std = torch.tensor([1.0, 3.0]).std(unbiased=True)
    lower = mean - FEDDMC_LIT_Z * std
    # The deterministic backdoor model makes new_grads far below the permitted
    # interval, so the released np.clip rule selects its lower boundary.
    expected_upload = -lower
    assert torch.allclose(client_states[2]["weight"], expected_upload.reshape(1))
    assert torch.equal(client_states[2]["weight"], client_states[3]["weight"])
    assert torch.equal(client_states[2]["counter"], global_state["counter"])

    metadata = attack_metadata("lit", config)
    assert metadata["lit_z"] == FEDDMC_LIT_Z == 0.48
    assert metadata["lit_backdoor_poison_ratio"] == FEDDMC_LIT_POISON_RATIO == 0.2
