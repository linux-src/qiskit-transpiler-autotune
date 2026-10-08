"""Benchmark circuits and target devices."""

from __future__ import annotations

import warnings
from pathlib import Path

from qiskit import QuantumCircuit, qpy
from qiskit_ibm_runtime import fake_provider

# Circuits whose layout cannot be found by VF2 alone, so routing actually matters.
SUITE: list[tuple[str, int]] = [
    ("qft", 8), ("qft", 16), ("qft", 32),
    ("qftentangled", 16), ("qftentangled", 32),
    ("qaoa", 8), ("qaoa", 16), ("qaoa", 32),
    ("graphstate", 16), ("graphstate", 32),
    ("vqe_two_local", 8), ("vqe_two_local", 16), ("vqe_two_local", 32),
    ("randomcircuit", 8), ("randomcircuit", 16),
    ("hhl", 16), ("hhl", 32),
    ("multiplier", 8), ("multiplier", 16),
    ("cdkm_ripple_carry_adder", 16), ("cdkm_ripple_carry_adder", 32),
    ("bv", 32), ("dj", 32),
    ("qwalk", 8),
]

_SIZES = (6, 8, 10, 12, 16, 20, 24, 28, 32, 40)
_MAX_SIZE = {
    "qwalk": 8, "randomcircuit": 20, "multiplier": 24, "rg_qft_multiplier": 20,
    "vqe_two_local": 32, "qv": 20,
}
_DATASET_FAMILIES = [
    "qft", "qftentangled", "qpeexact", "qpeinexact", "qaoa", "graphstate", "vqe_two_local",
    "vqe_real_amp", "randomcircuit", "hhl", "multiplier", "rg_qft_multiplier",
    "cdkm_ripple_carry_adder", "modular_adder", "full_adder", "bv", "dj", "ghz", "wstate",
    "qwalk", "qv",
]

# Larger and more varied set for learning which circuits are worth tuning. It also
# keeps circuits that VF2 maps without SWAPs, since recognising them is part of the task.
DATASET: list[tuple[str, int]] = [
    (family, size)
    for family in _DATASET_FAMILIES
    for size in _SIZES
    if size <= _MAX_SIZE.get(family, max(_SIZES))
    and not (family == "multiplier" and size % 4)
    and not (family == "rg_qft_multiplier" and size % 4)
]

SUITES = {"main": SUITE, "dataset": DATASET}

# Benchmarks whose generator draws random numbers and accepts a seed.
_SEEDED = {"graphstate"}
_SEED = 42

DEVICES = {
    "sherbrooke": fake_provider.FakeSherbrooke,
    "torino": fake_provider.FakeTorino,
}


def circuit_id(name: str, size: int) -> str:
    return f"{name}_{size}"


def generate(name: str, size: int) -> QuantumCircuit:
    if name == "qv":
        from qiskit.circuit.library import quantum_volume

        qc = quantum_volume(size, seed=_SEED).decompose()
        qc.measure_all()
        qc.name = circuit_id(name, size)
        return qc

    from mqt.bench import BenchmarkLevel, get_benchmark

    kwargs = {"seed": _SEED} if name in _SEEDED else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        qc = get_benchmark(name, BenchmarkLevel.INDEP, size, **kwargs)
    qc.name = circuit_id(name, size)
    return qc


def load_suite(cache_dir: Path, suite=SUITE) -> dict[str, QuantumCircuit]:
    """Generate the suite once and keep it as QPY so every run sees identical circuits."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    circuits = {}
    for name, size in suite:
        cid = circuit_id(name, size)
        path = cache_dir / f"{cid}.qpy"
        if not path.exists():
            with path.open("wb") as f:
                qpy.dump(generate(name, size), f)
        with path.open("rb") as f:
            circuits[cid] = qpy.load(f)[0]
    return circuits


def get_device(name: str):
    return DEVICES[name]()
