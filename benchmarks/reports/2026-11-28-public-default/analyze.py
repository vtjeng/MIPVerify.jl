"""Validate the public-API measurements and reuse the archived five-plot analysis.

Pass the reference analyze.py exported from evidence revision 53e17cf as the only
argument. README.md gives the command and reproduction environment.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE_SHA256 = "d123ed890624825fc96249c50c5180a0b15922daf1a916c5925655ac4e7b3fbb"
METHODS = ("baseline", "candidate")
NUMERIC = (
    "total_time_seconds",
    "main_solve_wall_time_seconds",
    "bound_solver_call_count",
)
DEFINITIVE = {
    "certified_no_adversarial_example",
    "adversarial_example_found_or_best_known",
}


def read_runs():
    data, observations, first, runs = {}, {}, {}, {}
    for method in METHODS:
        records = []
        for path in sorted((ROOT / "runs").glob(f"{method}-*.csv.gz")):
            match = re.fullmatch(rf"{method}-(\d+)-(\d+)\.csv\.gz", path.name)
            assert match, path
            frame = pd.read_csv(path, keep_default_na=False)
            assert not frame.empty and frame.sample_index.is_unique, path
            for key in NUMERIC:
                frame[key] = pd.to_numeric(frame[key], errors="raise")
                assert np.isfinite(frame[key]).all() and (frame[key] >= 0).all(), (path, key)
            assert (frame.main_solve_wall_time_seconds <= frame.total_time_seconds).all(), path
            assert frame.witness_available.equals(frame.perturbed_input_value.ne("")), path
            accepted = frame.semantic_outcome.eq("adversarial_example_found_or_best_known")
            assert accepted.equals(frame.witness_verified), path
            assert accepted.equals(frame.witness_verification.eq("public_api")), path
            assert frame.witness_verified.equals(frame.witness_target_verified & frame.witness_perturbation_verified), path
            for encoded in frame.loc[frame.witness_available, "perturbed_input_value"]:
                point = np.fromstring(encoded, sep=";")
                assert len(point) == 784 and np.isfinite(point).all(), path
            frame["pass"], frame["lane"] = int(match[1]), int(match[2])
            # The public result omits repair metadata. Drop that historical diagnostic below.
            frame["witness_repaired"] = pd.NA
            records.append(frame)
        assert records, method
        rows = pd.concat(records, ignore_index=True).sort_values(["pass", "lane", "sample_index"])
        assert not rows.duplicated(["pass", "sample_index"]).any(), method
        assert set(rows["pass"]) == {1, 2}, method
        for repeat, group in rows.groupby("pass"):
            assert sorted(group.sample_index) == list(range(1, 2001)), (method, repeat)
        runs[method] = rows
        observations[method] = 2
        first[method] = rows.groupby("sample_index", sort=True).first()
        data[method] = rows.groupby("sample_index")[list(NUMERIC)].mean()
        data[method]["outside_final_solve_seconds"] = (
            data[method].total_time_seconds - data[method].main_solve_wall_time_seconds
        )
    for image in range(1, 2001):
        verdicts = {
            outcome
            for method in METHODS
            for outcome in runs[method].loc[runs[method].sample_index == image, "semantic_outcome"]
            if outcome in DEFINITIVE
        }
        assert len(verdicts) <= 1, (image, "contradictory definitive outcomes", verdicts)
    return data, observations, first, runs


def main():
    reference = Path(sys.argv[1])
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == REFERENCE_SHA256, reference
    spec = importlib.util.spec_from_file_location("reference_analysis", reference)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    analysis.ROOT = ROOT
    data, observations, first, runs = read_runs()
    summary, specs = analysis.calculate(data, observations, first, runs)
    summary["returned_witnesses"].pop("repaired_sample_ids_by_pass")
    summary["configuration"] = {
        "Target LP": "public API with backward_screen=false",
        "Backward bounds": "public API with backward_screen=true (the default)",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    analysis.write_pairs(data, runs)
    analysis.write_tables(summary)
    analysis.write_metrics(summary)
    analysis.plot(data, summary, specs)
    print(json.dumps({"metrics": summary["metrics"], "repeat_noise_pct": summary["repeat_noise_pct"]}, indent=2))


if __name__ == "__main__":
    main()
