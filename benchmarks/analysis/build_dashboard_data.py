"""Build daily.json from the nightly archive. Uses only the Python standard library."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def daily_rows(archive):
    # Early runs share a date directory. On March 6, two tracking rows point
    # to the same file; its elapsed time identifies the run that was archived.
    tracking = defaultdict(list)
    with (archive / "tracking.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            tracking[row["date"], row.get("run_id") or ""].append(row)

    dates = {}
    for (date, run_id), candidates in sorted(tracking.items()):
        path = archive / "runs" / date / run_id / "benchmark_metrics.csv"
        with path.open(newline="") as file:
            (metrics,) = csv.DictReader(file)
        matches = [
            row
            for row in candidates
            if float(row["wall_clock_seconds"]) == float(metrics["wall_clock_seconds"])
        ]
        if not matches:
            raise ValueError(f"{path} does not match any tracking row")
        # The tracking table also contains historical per-sample quantiles.
        combined = matches[-1] | {key: value for key, value in metrics.items() if value}
        record = {"date": date, "commit_sha": combined["commit_sha"]}
        if run_id:
            record["run_id"] = run_id
        for key, value in combined.items():
            if not value:
                continue
            if key.startswith("num_"):
                record[key] = int(value)
            elif key.endswith("_seconds"):
                record[key] = float(value)
            elif key in (
                "adversarial_example_objective",
                "backward_screen",
                "tightening_algorithm",
                "julia_version",
                "benchmark_schema_version",
                "semantic_outcome_schema_version",
            ):
                record[key] = value
        # Timestamped run IDs sort chronologically within a UTC date.
        dates[date] = record
    if not dates:
        raise ValueError("Empty benchmark archive")
    return list(dates.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="benchmark-results checkout")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = daily_rows(args.archive)
    # One line per day makes the nightly commit diff readable.
    data = ",\n".join(json.dumps(row, separators=(",", ":"), allow_nan=False) for row in rows)
    args.output.write_text(f"[\n{data}\n]\n")
    print(f"Wrote {len(rows)} daily results through {rows[-1]['date']} to {args.output}")


if __name__ == "__main__":
    main()
