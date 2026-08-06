from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from torch.utils.data import DataLoader

from ..config import FedConfig


@dataclass
class PipelineContext:
    config: FedConfig
    task_name: str
    attack_name: str
    defense_name: str
    output_dir: Path
    config_files: Dict[str, str]
    applied_hyperparameters: Dict[str, object] = field(default_factory=dict)
    prepared_dataloaders: Optional[
        Tuple[List[DataLoader], DataLoader, DataLoader]
    ] = None
    task: Any = None
    device: Any = None
    client_loaders: List[DataLoader] = field(default_factory=list)
    validation_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    clients: List[Any] = field(default_factory=list)
    ground_truth: Any = None
    defense: Any = None
    batched_executor: Any = None
    round_idx: int = 0
    global_state: Dict[str, Any] = field(default_factory=dict)
    training_state: Dict[str, Any] = field(default_factory=dict)
    client_states: List[Dict[str, Any]] = field(default_factory=list)
    defense_result: Any = None
    evaluation: Dict[str, Any] = field(default_factory=dict)
    event: Dict[str, Any] = field(default_factory=dict)
    round_observer: Optional[Callable[[Dict[str, Any]], None]] = None
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    result_path: Optional[Path] = None


class Stage(Protocol):
    name: str

    def run(self, context: PipelineContext) -> PipelineContext:
        ...
