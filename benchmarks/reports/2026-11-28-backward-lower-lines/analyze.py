"""Summarize paired observations; reject partial cohorts or mismatched labels."""

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median, quantiles

RULES = ("adaptive", "parallel")
FIELDS = ("total_s", "screen_s", "mip_s", "lp_calls")
CERT = "certified_no_adversarial_example"
WITNESS = "adversarial_example_found_or_best_known"


def observations(folder, network, count):
    data = {}
    for file in sorted(folder.glob(f"{network}-*.csv*")):
        opener = gzip.open if file.suffix == ".gz" else open
        lane = int(file.name.split("-")[-1].split(".")[0])
        with opener(file, "rt") as f:
            for row in csv.DictReader(f):
                image, repeat = int(row["image"]), int(row["pass"])
                rule = row["lower"]
                key = (image, rule, repeat)
                assert row["network"] == network and (image - 1) % 5 == lane
                assert key not in data and 1 <= image <= count
                assert rule in RULES and repeat in (1, 2)
                for field in FIELDS:
                    row[field] = float(row[field])
                    assert math.isfinite(row[field]) and row[field] >= 0
                assert (
                    row["screen_s"] <= row["total_s"] and row["mip_s"] <= row["total_s"]
                )
                assert (row["outcome"] == WITNESS) == (row["verified"] == "true")
                if row["source"] == "original_input":
                    assert not row["bounds"] and row["screen_s"] == 0
                else:
                    row["bounds"] = tuple(map(float, row["bounds"].split(";")))
                    assert (
                        len(row["bounds"]) == 10 and row["bounds"].count(-math.inf) == 1
                    )
                    assert all(
                        math.isfinite(v) or v == -math.inf for v in row["bounds"]
                    )
                    assert (max(row["bounds"]) < 0) == (
                        row["source"] == "backward_bound"
                    )
                data[key] = row
    assert len(data) == 4 * count, (len(data), count)
    return data


def analyze(folder, network, count):
    data = observations(folder, network, count)
    images = range(1, count + 1)
    summaries = {}
    for rule in RULES:
        runs = []
        for rep in (1, 2):
            rows = [data[i, rule, rep] for i in images]
            runs.append(
                {
                    "phases": {f: sum(r[f] for r in rows) for f in FIELDS},
                    "routes": dict(Counter(r["source"] for r in rows)),
                    "outcomes": dict(Counter(r["outcome"] for r in rows)),
                    "rivals_excluded": sum(
                        sum(v < 0 for v in r["bounds"]) - 1 for r in rows if r["bounds"]
                    ),
                }
            )
        summaries[rule] = runs
    overlap = Counter()
    cert_ids = {rule: [] for rule in RULES}
    ratios = []
    differences = []
    for i in images:
        rows = [data[i, rule, rep] for rule in RULES for rep in (1, 2)]
        assert not {CERT, WITNESS} <= {r["outcome"] for r in rows}, i
        for rule in RULES:
            assert data[i, rule, 1]["bounds"] == data[i, rule, 2]["bounds"], (i, rule)
        a, p = rows[0]["bounds"], rows[2]["bounds"]
        assert bool(a) == bool(p)
        if a:
            assert a.index(-math.inf) == p.index(-math.inf)
            if max(a) < 0 <= max(p):
                cert_ids["adaptive"].append(i)
            if max(p) < 0 <= max(a):
                cert_ids["parallel"].append(i)
            overlap["either_per_rival_certificate"] += all(
                min(x, y) < 0 for x, y in zip(a, p)
            )
            for x, y in zip(a, p):
                if x != -math.inf:
                    overlap["compared_rivals"] += 1
                    overlap["parallel_tighter"] += y < x
                    overlap["adaptive_tighter"] += x < y
                    overlap["only_adaptive_excludes"] += x < 0 <= y
                    overlap["only_parallel_excludes"] += y < 0 <= x
        at, pt = (
            mean(data[i, rule, rep]["total_s"] for rep in (1, 2)) for rule in RULES
        )
        ratios.append(pt / at)
        differences.append((i, pt - at))
    totals = {rule: [r["phases"]["total_s"] for r in summaries[rule]] for rule in RULES}
    return {
        "network": network,
        "images": count,
        "runs": summaries,
        "runtime_change_pct": 100
        * (mean(totals["parallel"]) / mean(totals["adaptive"]) - 1),
        "repeat_noise_pct": max(
            100 * abs(x - y) / mean((x, y)) for x, y in totals.values()
        ),
        "per_repeat_change_pct": [
            100 * (p / a - 1) for a, p in zip(totals["adaptive"], totals["parallel"])
        ],
        "bounds": dict(overlap),
        "exclusive_certificates": cert_ids,
        "per_image_time_ratios_p10_median_p90": [
            quantiles(ratios, n=10)[0],
            median(ratios),
            quantiles(ratios, n=10)[-1],
        ],
        "faster_within_1pct_slower": [
            sum(r < 0.99 for r in ratios),
            sum(0.99 <= r <= 1.01 for r in ratios),
            sum(r > 1.01 for r in ratios),
        ],
        "largest_absolute_time_changes": sorted(
            differences, key=lambda item: abs(item[1]), reverse=True
        )[:5],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--network", choices=("WK17a", "RSL18a"))
    args = parser.parse_args()
    result = [
        analyze(args.folder, n, size)
        for n, size in (("WK17a", 2000), ("RSL18a", 100))
        if not args.network or n == args.network
    ]
    print(json.dumps(result, indent=2))
