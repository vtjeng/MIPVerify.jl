"""Compare the two reuse settings from one paired pass per MNIST test image."""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "reuse-comparison.csv").open()))
assert len(rows) == 4000
paired = defaultdict(dict)
for row in rows:
    image, reuse = int(row["sample_index"]), int(row["reuse"])
    assert reuse in (0, 1) and reuse not in paired[image]
    paired[image][reuse] = row
assert set(paired) == set(range(1, 2001))
assert all(set(p) == {0, 1} for p in paired.values())
assert all(p[0]["source"] == p[1]["source"] for p in paired.values())


def total(key, reuse):
    return math.fsum(float(p[reuse][key]) for p in paired.values())


a, b = total("total_time_seconds", 0), total("total_time_seconds", 1)
summary = {
    "scope": "One paired observation per image and reuse setting, all 2,000 images, same physical core. Order reverses between disjoint odd/even image groups.",
    "without_reuse_seconds": a,
    "with_reuse_seconds": b,
    "saved_seconds": a - b,
    "reduction_pct": 100 * (1 - b / a),
    "median_with_over_without_ratio": statistics.median(
        float(p[1]["total_time_seconds"]) / float(p[0]["total_time_seconds"])
        for p in paired.values()
    ),
    "main_solve_without_reuse_seconds": total("main_solve_wall_time_seconds", 0),
    "main_solve_with_reuse_seconds": total("main_solve_wall_time_seconds", 1),
    "bound_calls_without_reuse": total("bound_solver_call_count", 0),
    "bound_calls_with_reuse": total("bound_solver_call_count", 1),
}
rank = sorted(
    paired,
    key=lambda i: (
        float(paired[i][0]["total_time_seconds"])
        - float(paired[i][1]["total_time_seconds"])
    ),
    reverse=True,
)
summary["reduction_excluding_largest_savings_pct"] = {
    str(n): 100
    * (
        1
        - math.fsum(float(paired[i][1]["total_time_seconds"]) for i in rank[n:])
        / math.fsum(float(paired[i][0]["total_time_seconds"]) for i in rank[n:])
    )
    for n in [1, 2, 5, 10]
}
summary["top_savings"] = [
    {
        "sample": i,
        "saved_seconds": float(paired[i][0]["total_time_seconds"])
        - float(paired[i][1]["total_time_seconds"]),
    }
    for i in rank[:5]
]
summary["top_regressions"] = [
    {
        "sample": i,
        "saved_seconds": float(paired[i][0]["total_time_seconds"])
        - float(paired[i][1]["total_time_seconds"]),
    }
    for i in rank[-5:][::-1]
]
for key in ["semantic_outcome", "solve_status", "witness_verification"]:
    summary[key + "_counts"] = {
        str(mode): dict(Counter(p[mode][key] for p in paired.values()))
        for mode in [0, 1]
    }
    changes = defaultdict(list)
    for i, p in sorted(paired.items()):
        before, after = p[0][key], p[1][key]
        if before != after:
            changes[before, after].append(i)
    summary[key + "_changes"] = [
        dict(before=a, after=b, n=len(ids), samples=ids)
        for (a, b), ids in sorted(changes.items())
    ]
for mode in [0, 1]:
    for p in paired.values():
        r = p[mode]
        valid = r["semantic_outcome"] == "adversarial_example_found_or_best_known"
        assert bool(r["perturbed_input_value"]) == valid
        assert (r["witness_verification"] != "none") == valid
no_mip = [i for i, p in paired.items() if p[0]["source"] != "MIP"]
assert len(no_mip) == 1966
mip = [i for i in paired if i not in no_mip]
assert len(mip) == 34
summary["mip_images"] = mip
summary["without_reuse_mip_image_seconds"] = math.fsum(
    float(paired[i][0]["total_time_seconds"]) for i in mip
)
summary["with_reuse_mip_image_seconds"] = math.fsum(
    float(paired[i][1]["total_time_seconds"]) for i in mip
)
summary["repaired_samples_by_setting"] = {
    str(mode): [
        i
        for i, p in sorted(paired.items())
        if p[mode]["witness_repaired"].lower() == "true"
    ]
    for mode in [0, 1]
}
(ROOT / "reuse-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
