"""Tunable layout and routing configuration on top of the level 3 preset pass manager."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from qiskit.transpiler import PassManager, Target
from qiskit.transpiler.passes import (
    BarrierBeforeFinalMeasurements,
    SabreLayout,
    SabreSwap,
    SetLayout,
    VF2Layout,
)
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.transpiler.preset_passmanagers import common, generate_preset_pass_manager

_BARRIER_LABEL = "qiskit.transpiler.internal.routing.protection.barrier"


@dataclass(frozen=True)
class TuningConfig:
    """Layout and routing parameters.

    ``joint`` routing lets SabreLayout keep the routing it found while searching for
    the layout; ``separate`` routing discards it and reroutes with SabreSwap using
    ``heuristic``. Everything outside the layout and routing stages stays as in
    ``optimization_level=3``.
    """

    use_vf2: bool = True
    max_iterations: int = 4
    layout_trials: int = 20
    swap_trials: int = 20
    routing: Literal["joint", "separate"] = "joint"
    heuristic: Literal["basic", "lookahead", "decay"] = "decay"
    routing_trials: int = 20

    def as_dict(self) -> dict:
        return asdict(self)


O3_DEFAULT = TuningConfig()


def _vf2_match_not_found(property_set) -> bool:
    if property_set["layout"] is None:
        return True
    reason = property_set["VF2Layout_stop_reason"]
    return reason is not None and reason is not VF2LayoutStopReason.SOLUTION_FOUND


def _layout_stage(config: TuningConfig, target: Target, seed: int | None) -> PassManager:
    stage = PassManager()
    stage.append(SetLayout(None))
    if config.use_vf2:
        stage.append(
            ConditionalController(
                VF2Layout(seed=-1, call_limit=(30_000_000, 100_000), target=target),
                condition=lambda ps: not ps["layout"],
            )
        )
    stage.append(
        ConditionalController(
            [
                BarrierBeforeFinalMeasurements(_BARRIER_LABEL),
                SabreLayout(
                    target,
                    max_iterations=config.max_iterations,
                    seed=seed,
                    swap_trials=config.swap_trials,
                    layout_trials=config.layout_trials,
                    skip_routing=config.routing == "separate",
                ),
            ],
            condition=_vf2_match_not_found,
        )
    )
    embed = common.generate_embed_passmanager(target)
    stage.append(
        ConditionalController(
            embed.to_flow_controller(), condition=lambda ps: ps["final_layout"] is None
        )
    )
    return stage


def _routing_stage(config: TuningConfig, target: Target, seed: int | None) -> PassManager:
    vf2_call_limit, vf2_max_trials = common.get_vf2_limits(3, None, None)
    return common.generate_routing_passmanager(
        SabreSwap(target, heuristic=config.heuristic, seed=seed, trials=config.routing_trials),
        target,
        coupling_map=target.build_coupling_map(),
        vf2_call_limit=vf2_call_limit,
        vf2_max_trials=vf2_max_trials,
        seed_transpiler=-1,
        use_barrier_before_measurement=True,
    )


def build_pass_manager(config: TuningConfig, backend, seed: int | None = None) -> PassManager:
    """Level 3 preset pass manager with the layout and routing stages replaced."""
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed)
    target = backend.target
    pm.layout = _layout_stage(config, target, seed)
    if config.routing == "separate":
        pm.routing = _routing_stage(config, target, seed)
    return pm
