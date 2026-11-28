"""Validate and summarize the complete hosted public-API RSL18a comparison."""

import csv
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = {}
for path in sorted((ROOT / "runs").glob("*.csv.gz")):
    method, repeat, lane = path.name.removesuffix(".csv.gz").split("-")
    assert method in ("baseline", "candidate"), path
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            image = int(row["sample_index"])
            key = method, int(repeat), image
            assert key not in rows, key
            row["lane"] = int(lane)
            row["seconds"] = float(row["total_time_seconds"])
            assert row["seconds"] >= 0, key
            if row["witness_verified"] == "true":
                assert row["witness_available"] == "true", key
                assert row["witness_target_verified"] == "true", key
                assert row["witness_perturbation_verified"] == "true", key
                assert len(row["perturbed_input_value"].split(";")) == 784, key
            rows[key] = row

# The fixed experiment covers every selected image, both methods and both repeats.
expected = {
    (method, repeat, image)
    for method in ("baseline", "candidate")
    for repeat in (1, 2)
    for image in range(1, 101)
}
assert rows.keys() == expected, (len(rows), sorted(expected - rows.keys()))
summary = {"observations": len(rows), "samples": 100, "passes": {}, "transitions": []}
for method in ("baseline", "candidate"):
    for repeat in (1, 2):
        selected = [rows[method, repeat, image] for image in range(1, 101)]
        summary["passes"][f"{method}-{repeat}"] = {
            "seconds": sum(row["seconds"] for row in selected),
            "sources": dict(Counter(row["source"] for row in selected)),
            "statuses": dict(Counter(row["solve_status"] for row in selected)),
            "outcomes": dict(Counter(row["semantic_outcome"] for row in selected)),
            "verified_witnesses": sum(row["witness_verified"] == "true" for row in selected),
            "unverified_witnesses": sum(
                row["witness_available"] == "true" and row["witness_verified"] != "true"
                for row in selected
            ),
        }

pairs = []
for image in range(1, 101):
    paired = [rows[method, repeat, image] for method in ("baseline", "candidate") for repeat in (1, 2)]
    assert len({row["lane"] for row in paired}) == 1, image
    baseline = statistics.mean(rows["baseline", repeat, image]["seconds"] for repeat in (1, 2))
    candidate = statistics.mean(rows["candidate", repeat, image]["seconds"] for repeat in (1, 2))
    pairs.append({
        "sample_index": image,
        "baseline_seconds": baseline,
        "candidate_seconds": candidate,
        "candidate_over_baseline": candidate / baseline,
    })
    for repeat in (1, 2):
        before, after = rows["baseline", repeat, image], rows["candidate", repeat, image]
        if (before["solve_status"], before["semantic_outcome"]) != (after["solve_status"], after["semantic_outcome"]):
            summary["transitions"].append({
                "sample_index": image,
                "repeat": repeat,
                "baseline_status": before["solve_status"],
                "candidate_status": after["solve_status"],
                "baseline_outcome": before["semantic_outcome"],
                "candidate_outcome": after["semantic_outcome"],
                "candidate_source": after["source"],
            })
        # A solver time limit is unresolved. Definitive opposite verdicts are a regression.
        definitive = {before["semantic_outcome"], after["semantic_outcome"]}
        assert definitive != {"certified_no_adversarial_example", "adversarial_example_found_or_best_known"}, (image, repeat)

for method in ("baseline", "candidate"):
    totals = [summary["passes"][f"{method}-{repeat}"]["seconds"] for repeat in (1, 2)]
    summary[f"{method}_mean_seconds"] = statistics.mean(totals)
    summary[f"{method}_repeat_difference_percent"] = 100 * abs(totals[1] - totals[0]) / statistics.mean(totals)
summary["savings_percent"] = 100 * (1 - summary["candidate_mean_seconds"] / summary["baseline_mean_seconds"])
summary["noise_floor_percent"] = max(summary[f"{method}_repeat_difference_percent"] for method in ("baseline", "candidate"))
ratios = [pair["candidate_over_baseline"] for pair in pairs]
summary["median_ratio"] = statistics.median(ratios)
summary["faster_by_over_one_percent"] = sum(ratio < 0.99 for ratio in ratios)
summary["slower_by_over_one_percent"] = sum(ratio > 1.01 for ratio in ratios)
summary["within_one_percent"] = sum(0.99 <= ratio <= 1.01 for ratio in ratios)
deciles = statistics.quantiles(ratios, n=10, method="inclusive")
summary["ratio_quantiles"] = {
    "p10": deciles[0],
    "median": statistics.median(ratios),
    "p90": deciles[8],
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (ROOT / "paired-means.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=pairs[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(pairs)
print(json.dumps({key: value for key, value in summary.items() if key != "transitions"}, indent=2))
