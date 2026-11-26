"""Match actual timed points to exact-checked inputs and select the rest for replay.

Run without arguments against the archived timed-witnesses.csv.gz. To regenerate
that input from a benchmark runner's raw outputs, pass --raw-output /path/to/raw.
"""

import argparse
import csv
import gzip
import io
import json
from collections import Counter
from pathlib import Path


AUDIT = Path(__file__).resolve().parent


def extract_points(raw_output):
    points = []
    route_summary = {}
    with (AUDIT / "research-fallback-images.csv").open() as stream:
        expected_fallbacks = {int(row["sample_index"]) for row in csv.DictReader(stream)}
    for observation in (1, 2):
        route_counts = Counter()
        fallback_ids = set()
        seen = set()
        for lane in range(4):
            source = raw_output / f"candidate-{observation}-{lane}" / "benchmark_per_sample.csv"
            with source.open() as stream:
                for row in csv.DictReader(stream):
                    image = int(row["sample_index"])
                    assert image not in seen
                    seen.add(image)
                    route_counts[row["source"]] += 1
                    if row["source"] == "MIP":
                        fallback_ids.add(image)
                    if row["semantic_outcome"] == "adversarial_example_found_or_best_known":
                        points.append(
                            {
                                "sample_index": int(row["sample_index"]),
                                "pass": observation,
                                "route": row["source"],
                                "perturbed_input_value": row["perturbed_input_value"],
                            }
                        )
        assert seen == set(range(1, 2001))
        assert fallback_ids == expected_fallbacks
        route_summary[str(observation)] = {"source_counts": dict(route_counts), "fallback_image_ids": sorted(fallback_ids)}
    (AUDIT / "timed-route-summary.json").write_text(json.dumps(route_summary, indent=2) + "\n")
    points.sort(key=lambda row: (row["sample_index"], row["pass"]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(points[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(points)
    (AUDIT / "timed-witnesses.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))


def check_points():
    with (AUDIT / "confirmed-witnesses.csv").open() as stream:
        known = {int(row["sample_index"]): row for row in csv.DictReader(stream)}
    with gzip.open(AUDIT / "timed-witnesses.csv.gz", "rt") as stream:
        points = list(csv.DictReader(stream))
    assert len(points) == 2 * len(known)
    comparisons = []
    separate_replay = []
    for observation in (1, 2):
        seen = set()
        for row in points:
            if int(row["pass"]) != observation:
                continue
            image = int(row["sample_index"])
            assert image in known and image not in seen
            seen.add(image)
            same = row["perturbed_input_value"] == known[image]["perturbed_input_value"]
            comparisons.append([image, observation, row["route"], same])
            if not same or row["route"] == "MIP":
                separate_replay.append([image, observation, row["perturbed_input_value"]])
        assert seen == set(known)
    for name, header, rows in [
        (
            "timed-witness-crosscheck.csv",
            ["sample_index", "pass", "route", "identical_to_exact_checked_input"],
            comparisons,
        ),
        (
            "timed-mip-witnesses.csv",
            ["sample_index", "pass", "perturbed_input_value"],
            separate_replay,
        ),
    ]:
        with (AUDIT / name).open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(sorted(rows))
    print(
        "Timed witness records:", len(comparisons),
        "independent exact coordinate matches:", sum(row[-1] for row in comparisons),
        "points for separate replay:", len(separate_replay),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=Path)
    arguments = parser.parse_args()
    if arguments.raw_output is not None:
        extract_points(arguments.raw_output)
    check_points()
