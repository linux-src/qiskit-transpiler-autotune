"""Hiding unreliable couplers from the transpiler."""

from __future__ import annotations

from qiskit.transpiler import Target


def prune_target(target: Target, tau: float) -> Target:
    """Copy of ``target`` without the two-qubit couplings whose error is at least ``tau``.

    A coupler is dropped in both directions if either direction is bad, so the pruned
    coupling map stays symmetric where the original was. Couplings without calibrated
    error are kept. Everything else (qubits, single-qubit gates, measurements, control
    flow) is copied unchanged.
    """
    bad = set()
    for name in target.operation_names:
        for qargs, props in target[name].items():
            if qargs is None or len(qargs) != 2:
                continue
            if props is not None and props.error is not None and props.error >= tau:
                bad.add(frozenset(qargs))

    pruned = Target(
        description=target.description,
        num_qubits=target.num_qubits,
        dt=target.dt,
        granularity=target.granularity,
        min_length=target.min_length,
        pulse_alignment=target.pulse_alignment,
        acquire_alignment=target.acquire_alignment,
        qubit_properties=target.qubit_properties,
        concurrent_measurements=target.concurrent_measurements,
    )
    for name in target.operation_names:
        props = target[name]
        if None in props:
            pruned.add_instruction(target.operation_from_name(name), name=name)
            continue
        props = {q: p for q, p in props.items() if frozenset(q) not in bad}
        pruned.add_instruction(target.operation_from_name(name), props, name=name)
    return pruned
