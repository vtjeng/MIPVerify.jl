"""Reproduce alternating, CPU-pinned WK17a measurements in separate checkouts."""

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
CORES = (0, 2, 4, 6)


def run_lane(lane, first, last, rounds, output, trees):
    order = ("lp", "target_lp", "target_lp", "lp")
    if lane % 2:
        order = ("target_lp", "lp", "lp", "target_lp")
    for run, mode in enumerate(order[:rounds]):
        label = f"{first}-{last}-lane{lane}-run{run}-{mode}"
        destination = output / label
        if (destination / "benchmark_metrics.csv").exists():
            continue
        command = [
            "taskset", "-c", str(CORES[lane]), "julia", "--project=benchmarks",
            str(HERE / "driver.jl"), "--samples", f"{first + lane}:4:{last}",
            "--tightening", mode, "--out", str(destination),
        ]
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", JULIA_NUM_THREADS="1", OMP_NUM_THREADS="1")
        with (output / f"{label}.log").open("w") as log:
            subprocess.run(command, cwd=trees[mode], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(f"Completed {label}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=int)
    parser.add_argument("last", type=int)
    parser.add_argument("rounds", type=int, choices=[2, 4])
    args = parser.parse_args()
    output = Path(os.environ.get("WK17_OUTPUT", HERE / "runs")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trees = {"lp": os.environ["WK17_BASE_TREE"], "target_lp": os.environ["WK17_TARGET_TREE"]}
    with ThreadPoolExecutor(len(CORES)) as pool:
        futures = [pool.submit(run_lane, lane, args.first, args.last, args.rounds, output, trees)
                   for lane in range(len(CORES))]
        for future in futures:
            future.result()
