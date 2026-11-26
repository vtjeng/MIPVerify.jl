"""Compare LP and target-LP runs by image. Use 'all' to include the fresh cohort."""

import csv
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("WK17_OUTPUT", Path(__file__).resolve().parent / "runs"))
METHODS = ("lp", "target_lp")
DEFINITIVE = {"certified_no_adversarial_example", "adversarial_example_found_or_best_known"}


def read_cohort(first, last, rounds):
    images = defaultdict(lambda: defaultdict(list))
    for lane in range(4):
        order = ("lp", "target_lp", "target_lp", "lp")
        if lane % 2:
            order = ("target_lp", "lp", "lp", "target_lp")
        expected_images = list(range(first + lane, last + 1, 4))
        for run, mode in enumerate(order[:rounds]):
            directory = ROOT / f"{first}-{last}-lane{lane}-run{run}-{mode}"
            with (directory / "benchmark_per_sample.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            assert [int(row["sample_index"]) for row in rows] == expected_images
            for image, row in zip(expected_images, rows):
                images[image][mode].append(row)
    return images


def totals(pairs):
    return sum(a for _, a, _ in pairs), sum(b for _, _, b in pairs)


def reduction(pairs):
    baseline, candidate = totals(pairs)
    return 100 * (1 - candidate / baseline)


def report(images):
    pairs = []
    resolved_every_time = []
    for image, methods in sorted(images.items()):
        outcomes = {r["semantic_outcome"] for rows in methods.values() for r in rows}
        assert len(outcomes & DEFINITIVE) <= 1, (image, outcomes)
        baseline, candidate = [statistics.mean(float(r["total_time_seconds"]) for r in methods[m])
                               for m in METHODS]
        pair = (image, baseline, candidate)
        pairs.append(pair)
        if outcomes <= DEFINITIVE:
            resolved_every_time.append(pair)

    print(f"Images {min(images)}–{max(images)} ({len(images)} pairs)")
    a, b = totals(pairs)
    print(f"Paired totals: {a:.2f} and {b:.2f} seconds, {reduction(pairs):.2f}% reduction")
    for mode in METHODS:
        repeats = min(len(methods[mode]) for methods in images.values())
        for replicate in range(repeats):
            rows = [methods[mode][replicate] for methods in images.values()]
            total = sum(float(r["total_time_seconds"]) for r in rows)
            counts = Counter(r["semantic_outcome"] for r in rows)
            print(f"  {mode} replicate {replicate+1}: {total:.2f}s, {dict(counts)}")

    ordered = sorted(pairs, key=lambda p: p[1] - p[2], reverse=True)
    for n in (1, 5, 10):
        print(f"Excluding the {n} largest savings: {reduction(ordered[n:]):.2f}%")
    ratios = [b/a for _, a, b in pairs]
    print(f"Median ratio: {statistics.median(ratios):.3f}")
    print("Faster / within 1% / slower:", sum(r < .99 for r in ratios),
          sum(.99 <= r <= 1.01 for r in ratios), sum(r > 1.01 for r in ratios))
    print(f"Resolved every time: {len(resolved_every_time)} images, "
          f"{reduction(resolved_every_time):.2f}% reduction")
    rng = random.Random(20261125)
    gains = sorted(reduction(rng.choices(pairs, k=len(pairs))) for _ in range(4000))
    print(f"Paired-image bootstrap, 95% interval: {gains[100]:.2f}–{gains[-101]:.2f}%")


if __name__ == "__main__":
    first = read_cohort(1, 500, 4)
    report(first)
    report({i: rows for i, rows in first.items() if i > 100})
    if "all" in sys.argv:
        fresh = read_cohort(501, 2000, 2)
        report(fresh)
        report(first | fresh)
