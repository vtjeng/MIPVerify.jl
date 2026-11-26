"""Isolate ReLU bound reuse while keeping the backward screen and search identical."""

import argparse
import csv
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--core", type=int, default=8)
args = parser.parse_args()
raw = args.raw.resolve()
raw.mkdir(parents=True, exist_ok=True)
env = dict(
    os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
)
rows = []
for first, reuse in [(1, False), (1, True), (2, True), (2, False)]:
    output = raw / f"from-{first}-reuse-{str(reuse).lower()}"
    assert not output.exists(), f"Refusing to replace an existing timed run: {output}"
    print("Starting", output.name, flush=True)
    with (raw / f"{output.name}.log").open("w") as log:
        subprocess.run(
            [
                "taskset",
                "-c",
                str(args.core),
                "julia",
                "--project=benchmarks",
                "benchmarks/backward_margin_screen.jl",
                "--samples",
                f"{first}:2:2000",
                "--prebound-fallback",
                str(reuse).lower(),
                "--out",
                str(output),
            ],
            cwd=args.candidate,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with (output / "benchmark_per_sample.csv").open() as f:
        observed = list(csv.DictReader(f))
    assert [int(r["sample_index"]) for r in observed] == list(range(first, 2001, 2))
    rows.extend(dict(r, reuse=int(reuse)) for r in observed)
fields = [
    "sample_index",
    "reuse",
    "source",
    "solve_status",
    "semantic_outcome",
    "total_time_seconds",
    "main_solve_wall_time_seconds",
    "bound_solver_call_count",
]
with args.output.open("w") as f:
    fields = list(dict.fromkeys([*fields, *rows[0]]))
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: (int(r["sample_index"]), r["reuse"])))
