from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..defenses.base import DefenseContext
from ..utils import clip_client_updates
from .contracts import PipelineContext, Stage


@dataclass(frozen=True)
class ConfigStage:
    name: str = "config"

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg = context.config
        identities = {
            "task": (cfg.task_name, context.task_name),
            "attack": (cfg.attack_type, context.attack_name),
            "defense": (cfg.defense_type, context.defense_name),
        }
        mismatched = [
            label for label, (configured, selected) in identities.items()
            if configured != selected
        ]
        if mismatched:
            raise ValueError(
                "Pipeline identity does not match FedConfig: " + ", ".join(mismatched)
            )
        return context


@dataclass(frozen=True)
class DataStage:
    build: Callable
    name: str = "data"

    def run(self, context: PipelineContext) -> PipelineContext:
        prepared = context.prepared_dataloaders
        if prepared is None:
            prepared = self.build(context.config)
            context.prepared_dataloaders = prepared
        context.client_loaders, context.validation_loader, context.test_loader = prepared
        if len(context.client_loaders) != context.config.num_clients:
            raise ValueError(
                "Prepared dataloader count does not match config.num_clients: "
                f"{len(context.client_loaders)} != {context.config.num_clients}"
            )
        if context.validation_loader is None:
            raise ValueError("A clean server validation loader is required.")
        return context


@dataclass(frozen=True)
class ClientStage:
    build: Callable
    name: str = "clients"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.clients, context.ground_truth = self.build(
            context.config,
            context.device,
            context.client_loaders,
            context.task,
        )
        return context


@dataclass(frozen=True)
class ClientTrainStage:
    batched_runner: Callable
    name: str = "client_train"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.defense is None:
            raise RuntimeError("Defense must be initialized before client training")
        context.global_state = context.defense.state_dict_for_clients()
        if context.device.type == "cuda":
            context.training_state = {
                key: value.to(context.device, non_blocking=True)
                for key, value in context.global_state.items()
            }
        else:
            context.training_state = context.global_state
        if context.batched_executor is None:
            context.client_states = [
                client.local_step(
                    context.training_state,
                    reference_state_dict=context.global_state,
                )
                for client in context.clients
            ]
        else:
            context.client_states = self.batched_runner(
                context.clients,
                context.training_state,
                context.global_state,
                context.batched_executor,
            )
        context.training_state = {}
        return context


@dataclass(frozen=True)
class AttackStage:
    apply: Callable
    name: str = "attack"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.defense is None:
            raise RuntimeError("Defense must be initialized before coordinated attacks")
        # Paper-defined omniscient attacks operate on trainable gradients, not
        # BatchNorm running statistics or other state-dict buffers.
        parameter_names = tuple(
            name
            for name, parameter in context.defense.global_model.named_parameters()
            if parameter.requires_grad
        )
        self.apply(
            context.config,
            context.defense_name,
            context.global_state,
            context.client_states,
            parameter_names,
            context.clients,
        )
        return context


@dataclass(frozen=True)
class UploadClipStage:
    """Apply the common post-attack client-update clipping boundary."""

    name: str = "upload_clip"

    def run(self, context: PipelineContext) -> PipelineContext:
        states, stats = clip_client_updates(
            context.client_states,
            context.global_state,
            max_norm=context.config.client_update_clip,
        )
        context.client_states = states
        context.upload_clip_stats = stats
        return context


@dataclass(frozen=True)
class DefenseStage:
    name: str = "defense"

    def run(self, context: PipelineContext) -> PipelineContext:
        defense_context = DefenseContext(
            round_idx=context.round_idx,
            global_state=context.global_state,
            client_states=context.client_states,
        )
        context.defense_result = context.defense.aggregate(defense_context)
        return context


@dataclass(frozen=True)
class EvaluationStage:
    evaluate: Callable
    build_event: Callable
    evaluate_extra: Callable | None = None
    name: str = "evaluation"

    def run(self, context: PipelineContext) -> PipelineContext:
        evaluated = self.evaluate(
            context.defense.global_model,
            context.test_loader,
            context.device,
            use_amp=context.config.use_amp,
            channels_last=context.config.channels_last,
        )
        if isinstance(evaluated, dict):
            context.evaluation = dict(evaluated)
        else:
            accuracy, correct, total = evaluated
            context.evaluation = {
                "accuracy": float(accuracy),
                "correct": int(correct),
                "total": int(total),
            }
        if self.evaluate_extra is not None:
            context.evaluation.update(self.evaluate_extra(context))
        context.event = self.build_event(context)
        return context


@dataclass(frozen=True)
class OutputStage:
    emit_console: Callable
    name: str = "output"

    def run(self, context: PipelineContext) -> PipelineContext:
        self.emit_console(context.event)
        if context.round_observer is not None:
            context.round_observer(context.event)
        return context


class RoundPipeline:
    """Run one ordered set of independent stages for every communication round."""

    def __init__(self, *stages: Stage) -> None:
        self.stages = stages

    def run(self, context: PipelineContext) -> PipelineContext:
        for round_idx in range(1, int(context.config.total_rounds) + 1):
            context.round_idx = round_idx
            for stage in self.stages:
                stage.run(context)
        return context


__all__ = [
    "AttackStage",
    "ClientStage",
    "ClientTrainStage",
    "ConfigStage",
    "DataStage",
    "DefenseStage",
    "EvaluationStage",
    "OutputStage",
    "RoundPipeline",
    "UploadClipStage",
]
