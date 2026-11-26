"""Archive the actual adversarial inputs returned during both timed passes."""

import argparse
import csv
import gzip
import io
from pathlib import Path

AUDIT = Path(__file__).resolve().parent


def extract(raw_output, output, ablation):
    points = []
    sources = []
    if ablation:
        for first in (1, 2):
            for reuse in ("false", "true"):
                path = raw_output / f"from-{first}-reuse-{reuse}" / "benchmark_per_sample.csv"
                sources.append((path, 1, reuse, list(range(first, 2001, 2))))
    else:
        for observation in (1, 2):
            for lane in range(4):
                path = raw_output / f"candidate-{observation}-{lane}" / "benchmark_per_sample.csv"
                sources.append((path, observation, "true", list(range(1 + lane, 2001, 4))))
    for source, observation, reuse, expected_ids in sources:
        with source.open() as stream:
            records = list(csv.DictReader(stream))
            assert [int(row["sample_index"]) for row in records] == expected_ids
            for row in records:
                if row["semantic_outcome"] != "adversarial_example_found_or_best_known":
                    continue
                assert row["perturbed_input_value"], (source, row["sample_index"])
                points.append(
                    {
                        "sample_index": int(row["sample_index"]),
                        "pass": observation,
                        "run": source.parent.name,
                        "reuse_bounds": reuse,
                        "route": row["source"],
                        "witness_verification": row["witness_verification"],
                        "witness_repaired": row["witness_repaired"],
                        "perturbed_input_value": row["perturbed_input_value"],
                    }
                )
    points.sort(key=lambda row: (row["sample_index"], row["run"]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(points[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(points)
    output.write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    print("Archived", len(points), "actual returned inputs at", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    name = "ablation-timed-witnesses.csv.gz" if arguments.ablation else "timed-witnesses.csv.gz"
    extract(arguments.raw_output, arguments.out or AUDIT / name, arguments.ablation)
