"""Time two WK17a verifiers twice on the same images and physical cores."""

import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "2026-11-25-wk17a-target-lp" / "driver.jl"
CORE = (0, 2, 4, 6)
FIELDS = (
    "sample_index",
    "solve_status",
    "semantic_outcome",
    "source",
    "total_time_seconds",
    "main_solve_wall_time_seconds",
    "bound_solver_call_count",
    "formulation_time_seconds",
    "bound_solver_wall_time_seconds",
    "bound_tightening_time_seconds",
)


def run(lane, step, raw, runs, trees):
    order = ("baseline", "candidate", "candidate", "baseline")
    if lane % 2:
        order = ("candidate", "baseline", "baseline", "candidate")
    method = order[step]
    observation = order[: step + 1].count(method)
    label = f"{method}-{observation}-{lane}"
    output = raw / label
    assert not output.exists(), f"Refusing to replace an existing timed run: {output}"
    assert not (runs / f"{label}.csv").exists(), f"Existing normalized run: {label}"
    ids = list(range(1 + lane, 2001, len(CORE)))
    samples = f"{ids[0]}:{len(CORE)}:2000"
    if method == "baseline":
        args = [
            str(DRIVER),
            "--samples",
            samples,
            "--tightening",
            "target_lp",
            "--out",
            str(output),
        ]
    else:
        args = [
            "benchmarks/backward_margin_screen.jl",
            "--samples",
            samples,
            "--out",
            str(output),
        ]
    env = dict(
        os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    print("Starting", label, flush=True)
    with (raw / f"{label}.log").open("w") as log:
        subprocess.run(
            ["taskset", "-c", str(CORE[lane]), "julia", "--project=benchmarks", *args],
            cwd=trees[method],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with (output / "benchmark_per_sample.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["sample_index"]) for r in rows] == ids
    tmp = runs / f"{label}.tmp"
    with tmp.open("w") as f:
        fields = list(dict.fromkeys([*FIELDS, *rows[0]]))
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            if method == "baseline":
                row["source"] = (
                    "unperturbed"
                    if row["solve_status"] == "SKIPPED_PREDICTED_IN_TARGETED"
                    else "MIP"
                )
                if row["source"] == "unperturbed":
                    row.update(
                        main_solve_wall_time_seconds=0, bound_solver_call_count=0
                    )
            writer.writerow(row)
    tmp.replace(runs / f"{label}.csv")
    print("Completed", label, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, required=True)
    cli.add_argument("--raw", type=Path, required=True)
    cli.add_argument("--runs", type=Path, required=True)
    cli.add_argument("--lanes", type=int, nargs="+", choices=range(4), default=range(4))
    cli.add_argument("--steps", type=int, nargs="+", choices=range(4), default=range(4))
    args = cli.parse_args()
    trees = dict(baseline=args.baseline.resolve(), candidate=args.candidate.resolve())
    raw, runs = args.raw.resolve(), args.runs.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    def series(lane):
        for step in args.steps:
            run(lane, step, raw, runs, trees)

    with ThreadPoolExecutor(len(args.lanes)) as pool:
        list(pool.map(series, args.lanes))


if __name__ == "__main__":
    main()
