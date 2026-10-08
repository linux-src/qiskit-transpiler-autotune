"""Running transpilation experiments and recording results."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qta.metrics import measure

_PACKAGES = ["qiskit", "qiskit-ibm-runtime", "mqt.bench", "optuna", "qiskit-transpiler-autotune"]


def evaluate(circuit: QuantumCircuit, pm: PassManager, target) -> dict:
    start = time.perf_counter()
    out = pm.run(circuit)
    elapsed = time.perf_counter() - start
    return {**measure(out, target).as_dict(), "time_s": elapsed}


def environment() -> dict:
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {p: version(p) for p in _PACKAGES},
    }


class CsvLog:
    """Append-only CSV that can resume an interrupted run."""

    def __init__(self, path: Path, fields: list[str], key: list[str]):
        self.path = Path(path)
        self.fields = fields
        self.key = key
        self.done: set[tuple] = set()
        if self.path.exists():
            with self.path.open() as f:
                self.done = {tuple(row[k] for k in key) for row in csv.DictReader(f)}
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fields).writeheader()

    def has(self, **key) -> bool:
        return tuple(str(key[k]) for k in self.key) in self.done

    def write(self, row: dict) -> None:
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, self.fields, extrasaction="ignore").writerow(row)
        self.done.add(tuple(str(row[k]) for k in self.key))


def write_environment(out_dir: Path, **extra) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(
        json.dumps({**environment(), **extra}, indent=2, ensure_ascii=False, default=str)
    )
