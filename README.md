# qiskit-transpiler-autotune

Automatic per-circuit tuning of Qiskit transpiler passes (layout & routing) to reduce
two-qubit gate count and depth on noisy IBM devices.

## Setup

```bash
uv venv .venv
uv pip install -e ".[dev,sim]"
```

## Tests

```bash
.venv/bin/pytest
```

## License

Apache-2.0
