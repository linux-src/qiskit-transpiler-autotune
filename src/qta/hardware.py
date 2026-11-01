"""Does ESP rank compilations of a circuit the way a real IBM QPU does?

    python -m qta.hardware submit --backend ibm_fez --out results/hw/run1 --circuits-dir results/circuits
    python -m qta.hardware send --out results/hw/run1
    python -m qta.hardware collect --out results/hw/run1

``submit`` compiles every small circuit with level 3 and many seeds against the live
calibration of the backend, keeps circuits whose ESP differs noticeably between seeds,
and plans a few compilations per circuit (spread over the ESP range) for one Sampler job;
``send`` submits them.
``collect`` fetches the counts and compares ESP with the Hellinger fidelity to the ideal
output distribution. With ``--dry-run`` the backend is a fake snapshot and the job runs
locally on Aer, so the whole pipeline can be checked without spending QPU time.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import hellinger_fidelity
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from qta.circuits import DATASET, get_device, load_suite
from qta.esp_check import ideal_distribution, rank_agreement, with_measurements
from qta.metrics import measure
from qta.runner import write_environment


def compile_seeds(qc: QuantumCircuit, backend, seeds: int) -> tuple[pd.DataFrame, list]:
    rows, outs = [], []
    for seed in range(seeds):
        out = generate_preset_pass_manager(3, backend=backend, seed_transpiler=seed).run(qc)
        m = measure(out, backend.target)
        rows.append({"seed": seed, "n_2q": m.n_2q, "depth": m.depth, "log_esp": m.log_esp,
                     "esp": m.esp})
        outs.append(out)
    return pd.DataFrame(rows), outs


def spread_seeds(seeds: pd.DataFrame, k: int) -> list[int]:
    """Seeds at evenly spaced ESP quantiles, plus the seed with fewest two-qubit gates."""
    ordered = seeds.sort_values(["esp", "seed"]).reset_index(drop=True)
    picks = ordered.seed[np.linspace(0, len(ordered) - 1, k).round().astype(int)].tolist()
    picks.append(int(seeds.sort_values(["n_2q", "seed"]).seed.iloc[0]))
    return list(dict.fromkeys(int(s) for s in picks))


def single_creg(qc: QuantumCircuit) -> bool:
    return len(qc.cregs) == 1 and qc.cregs[0].size == qc.num_clbits


def qpu_seconds(circuits: list[QuantumCircuit], backend, shots: int) -> float:
    """Rough lower bound: shots times repetition delay plus circuit duration."""
    rep_delay = getattr(backend, "default_rep_delay", None) or 250e-6
    dt = backend.target.dt or 0.0
    total = 0.0
    for qc in circuits:
        duration = qc.estimate_duration(backend.target) if dt else 0.0
        total += shots * (rep_delay + duration)
    return total


def submit(args) -> None:
    if args.dry_run:
        backend = get_device(args.backend)
    else:
        backend = QiskitRuntimeService().backend(args.backend)
    suite = [(name, size) for name, size in DATASET if size <= args.max_qubits]
    circuits = load_suite(args.circuits_dir, suite)

    plan, compiled, candidates = [], [], []
    for cid, qc in circuits.items():
        qc = with_measurements(qc)
        if not single_creg(qc):
            continue
        seeds, outs = compile_seeds(qc, backend, args.seeds)
        seeds.insert(0, "circuit", cid)
        candidates.append(seeds)
        median = seeds.esp.median()
        ratio = seeds.esp.max() / max(seeds.esp.min(), 1e-300)
        print(f"{cid}: median ESP {median:.3f}, best/worst {ratio:.2f}", flush=True)
        if not args.min_esp <= median <= args.max_esp or ratio < args.min_ratio:
            continue
        plan.append((ratio, cid, seeds, outs))

    # Circuits with the widest ESP spread across seeds are the ones where the choice of
    # seed can matter on hardware at all.
    plan.sort(key=lambda p: -p[0])
    rows, ideal = [], {}
    for _, cid, seeds, outs in plan[:args.max_circuits]:
        ideal[cid] = ideal_distribution(with_measurements(circuits[cid]))
        for seed in spread_seeds(seeds, args.per_circuit):
            r = seeds[seeds.seed == seed].iloc[0]
            rows.append({"pub": len(compiled), "circuit": cid, "seed": seed, "n_2q": int(r.n_2q),
                         "depth": int(r.depth), "log_esp": r.log_esp, "esp": r.esp})
            compiled.append(outs[seed])

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(candidates).to_csv(out_dir / "candidates.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "plan.csv", index=False)
    (out_dir / "ideal.json").write_text(json.dumps(ideal))
    with (out_dir / "submitted.qpy").open("wb") as f:
        qpy.dump(compiled, f)
    estimate = qpu_seconds(compiled, backend, args.shots)
    print(f"{len(plan)} circuits pass the filter, {len(compiled)} compilations to run, "
          f"~{estimate:.0f} s of shots (without job overhead)")
    write_environment(out_dir, experiment="hardware", backend=backend.name, args=vars(args),
                      estimated_qpu_s=estimate)
    if hasattr(backend, "properties") and not args.dry_run:
        (out_dir / "calibration.json").write_text(
            json.dumps(backend.properties().to_dict(), default=str))

    if args.yes:
        send(out_dir, backend, args.shots, args.dry_run)
    else:
        print(f"review {out_dir / 'plan.csv'}, then: python -m qta.hardware send --out {out_dir}")


def send(out_dir: Path, backend, shots: int, dry_run: bool = False) -> None:
    with (out_dir / "submitted.qpy").open("rb") as f:
        compiled = qpy.load(f)
    job = SamplerV2(mode=backend).run(compiled, shots=shots)
    meta = {"job_id": job.job_id(), "backend": backend.name, "shots": shots,
            "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dry_run": dry_run}
    (out_dir / "job.json").write_text(json.dumps(meta, indent=2))
    print(f"job {meta['job_id']} submitted")
    if dry_run:
        evaluate(out_dir, job.result())


def send_planned(args) -> None:
    env = json.loads((args.out / "environment.json").read_text())
    planned = env["args"]
    if planned["dry_run"]:
        backend = get_device(planned["backend"])
    else:
        backend = QiskitRuntimeService().backend(planned["backend"])
    send(args.out, backend, planned["shots"], planned["dry_run"])


def evaluate(out_dir: Path, result) -> None:
    plan = pd.read_csv(out_dir / "plan.csv")
    with (out_dir / "submitted.qpy").open("rb") as f:
        compiled = qpy.load(f)
    ideal = json.loads((out_dir / "ideal.json").read_text())
    fidelities = []
    for row in plan.itertuples():
        qc = compiled[row.pub]
        counts = getattr(result[row.pub].data, qc.cregs[0].name).get_counts()
        fidelities.append(hellinger_fidelity(ideal[row.circuit], counts))
    runs = plan.assign(fidelity=fidelities, device=json.loads(
        (out_dir / "job.json").read_text())["backend"])
    runs.to_csv(out_dir / "runs.csv", index=False)
    agreement = rank_agreement(runs)
    agreement.to_csv(out_dir / "rank_agreement.csv", index=False)
    print(runs[["circuit", "seed", "n_2q", "esp", "fidelity"]].to_string(index=False))
    print(agreement.to_string(index=False))


def collect(args) -> None:
    meta = json.loads((args.out / "job.json").read_text())
    job = QiskitRuntimeService().job(meta["job_id"])
    print(f"job {meta['job_id']}: {job.status()}")
    if job.status() != "DONE":
        return
    (args.out / "usage.json").write_text(json.dumps(job.metrics(), indent=2, default=str))
    evaluate(args.out, job.result())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit")
    p.add_argument("--backend", required=True, help="IBM backend name, or a fake one with --dry-run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--circuits-dir", type=Path, required=True)
    p.add_argument("--max-qubits", type=int, default=10)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--per-circuit", type=int, default=5, help="compilations per circuit by ESP quantile")
    p.add_argument("--max-circuits", type=int, default=10)
    p.add_argument("--min-esp", type=float, default=0.05)
    p.add_argument("--max-esp", type=float, default=0.95)
    p.add_argument("--min-ratio", type=float, default=1.1, help="best/worst ESP across seeds")
    p.add_argument("--shots", type=int, default=4000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true", help="send the job right after planning")
    p.set_defaults(func=submit)

    t = sub.add_parser("send", help="send a job planned by submit")
    t.add_argument("--out", type=Path, required=True)
    t.set_defaults(func=send_planned)

    c = sub.add_parser("collect")
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=collect)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
