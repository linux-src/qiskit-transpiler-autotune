# qiskit-transpiler-autotune

Automatic per-circuit tuning of Qiskit transpiler passes (layout & routing) to reduce
two-qubit gate count and depth on noisy IBM devices.

## Setup

```bash
uv venv .venv
uv pip install -e ".[dev,sim]"
```

## Experiments

```bash
# preset levels 0-3, 20 transpiler seeds each
.venv/bin/python -m qta.baseline --out results/baseline --seeds 20

# random search over layout/routing parameters, 40 configurations per circuit
.venv/bin/python -m qta.random_search --out results/random --trials 40

# comparison at an equal compile-time budget, tables and figures
.venv/bin/python -m qta.analysis --baseline results/baseline --search results/random --out results/report
```

Benchmark circuits come from MQT Bench and are cached as QPY in `<out>/circuits`
(or `--circuits-dir`), so all runs see identical circuits. Devices are the
`FakeSherbrooke` and `FakeTorino` snapshots from `qiskit-ibm-runtime`.
Each run also writes `environment.json` with package versions and the machine.
Run experiments one at a time: compile time is part of the comparison.

## Tests

```bash
.venv/bin/pytest
```

## License

Apache-2.0
