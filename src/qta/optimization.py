"""Tunable optimization stage on top of the level 3 preset pass manager."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ApplyLayout,
    CommutativeCancellation,
    ConsolidateBlocks,
    ContractIdleWiresInControlFlow,
    Depth,
    FixedPoint,
    GatesInBasis,
    InverseCancellation,
    MinimumPoint,
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
    Size,
    TwoQubitPeepholeOptimization,
    UnitarySynthesis,
    VF2PostLayout,
)
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import common, generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager

_PREFIX = "optimization_loop"


@dataclass(frozen=True)
class OptimizationConfig:
    """Which passes the optimization loop runs and how it decides to stop.

    The defaults reproduce ``optimization_level=3``. ``inverse_cancellation`` and
    ``consolidate`` (block collection followed by unitary resynthesis) are not part of
    level 3 in this Qiskit version and are offered as extra options. Every option
    preserves the circuit unitary (``approximation_degree`` stays 1).
    """

    peephole: bool = True
    remove_identity: bool = True
    optimize_1q: bool = True
    commutative_cancellation: bool = True
    inverse_cancellation: bool = False
    consolidate: bool = False
    loop: Literal["minimum_point", "fixed_point", "once"] = "minimum_point"
    vf2_post_layout: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


O3_OPTIMIZATION = OptimizationConfig()


def sample(rng: np.random.Generator) -> OptimizationConfig:
    return OptimizationConfig(
        peephole=bool(rng.integers(2)),
        remove_identity=bool(rng.integers(2)),
        optimize_1q=bool(rng.integers(2)),
        commutative_cancellation=bool(rng.integers(2)),
        inverse_cancellation=bool(rng.integers(2)),
        consolidate=bool(rng.integers(2)),
        loop=str(rng.choice(["minimum_point", "fixed_point", "once"])),
        vf2_post_layout=bool(rng.integers(2)),
    )


def _loop_check(loop: str):
    if loop == "minimum_point":
        setup = [Size(recurse=True), Depth(recurse=True), MinimumPoint(["depth", "size"], _PREFIX)]
        return setup, lambda ps: not ps[f"{_PREFIX}_minimum_point"]
    if loop == "fixed_point":
        setup = [Size(recurse=True), Depth(recurse=True), FixedPoint("size"), FixedPoint("depth")]
        return setup, lambda ps: not (ps["depth_fixed_point"] and ps["size_fixed_point"])
    return [], lambda ps: False


def optimization_stage(config: OptimizationConfig, pm_config: PassManagerConfig) -> PassManager:
    target = pm_config.target
    approx = pm_config.approximation_degree
    loop = []
    if config.consolidate:
        loop += [ConsolidateBlocks(target=target, approximation_degree=approx),
                 UnitarySynthesis(target=target, approximation_degree=approx)]
    if config.peephole:
        loop.append(TwoQubitPeepholeOptimization(target, approximation_degree=approx))
    if config.remove_identity:
        loop.append(RemoveIdentityEquivalent(approximation_degree=approx, target=target))
    if config.optimize_1q:
        loop.append(Optimize1qGatesDecomposition(basis=pm_config.basis_gates, target=target))
    if config.inverse_cancellation:
        loop.append(InverseCancellation())
    if config.commutative_cancellation:
        loop.append(CommutativeCancellation(target=target))
    loop.append(ContractIdleWiresInControlFlow())

    translation = PassManagerStagePluginManager().get_passmanager_stage(
        "translation", pm_config.translation_method or "default", pm_config, optimization_level=3
    )
    unroll = [
        GatesInBasis(pm_config.basis_gates, target=target),
        ConditionalController(
            translation.to_flow_controller(), condition=lambda ps: not ps["all_gates_in_basis"]
        ),
    ]

    setup, continue_loop = _loop_check(config.loop)
    stage = PassManager()
    stage.append(setup)
    stage.append(DoWhileController(loop + unroll + setup, do_while=continue_loop))
    if config.vf2_post_layout:
        call_limit, max_trials = common.get_vf2_limits(3, None, None, exact_match=True)
        stage.append([
            VF2PostLayout(target=target, seed=-1, call_limit=call_limit, max_trials=max_trials,
                          strict_direction=True),
            ConditionalController(ApplyLayout(), condition=common._apply_post_layout_condition),
        ])
    return stage


def build_pass_manager(config: OptimizationConfig, backend, seed: int | None = None,
                       target=None) -> PassManager:
    """Level 3 preset pass manager with the optimization stage replaced."""
    target = target or backend.target
    pm = generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=seed)
    pm_config = PassManagerConfig(
        target=target,
        basis_gates=list(target.operation_names),
        approximation_degree=1.0,
        seed_transpiler=seed,
    )
    pm.optimization = optimization_stage(config, pm_config)
    return pm
