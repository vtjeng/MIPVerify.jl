"""Verify final audit coverage against the completed raw benchmark observations."""

import csv
import hashlib
import gzip
import json
import re
import sys
from pathlib import Path


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def check(network, source, audit):
    expected_images = {"WK17a": 2000, "RSL18a": 100}[network]
    observations = {}
    for path in sorted(source.glob("*.csv*")):
        match = re.fullmatch(
            r"(baseline|candidate)-([12])-\d+\.csv(?:\.gz)?", path.name
        )
        if not match:
            continue
        method, repeat = match.groups()
        for row in read_rows(path):
            key = (method, int(repeat), int(row["sample_index"]))
            assert key not in observations, f"Duplicate observation {key}"
            observations[key] = row
    expected = {
        (method, repeat, image)
        for method in ("baseline", "candidate")
        for repeat in (1, 2)
        for image in range(1, expected_images + 1)
    }
    assert set(observations) == expected, "Incomplete or unexpected observation keys"
    definitive = {
        "certified_no_adversarial_example",
        "adversarial_example_found_or_best_known",
    }
    contradictory = []
    for image in range(1, expected_images + 1):
        outcomes = {
            row["semantic_outcome"]
            for (_, _, sample), row in observations.items()
            if sample == image and row["semantic_outcome"] in definitive
        }
        if len(outcomes) > 1:
            contradictory.append(image)
    assert not contradictory, (
        f"Opposing definitive verdicts across methods or repeats: {contradictory}"
    )
    witnesses = {
        key: row
        for key, row in observations.items()
        if row["witness_available"] == "true"
    }
    replay = {}
    for row in read_rows(audit / "witness-replay.csv"):
        key = (row["method"], int(row["observation"]), int(row["sample_index"]))
        assert key not in replay, f"Duplicate witness audit {key}"
        assert row["network"] == network
        replay[key] = row
    assert set(replay) == set(witnesses), (
        "Witness audit does not cover every saved point"
    )
    for key, row in witnesses.items():
        digest = hashlib.sha256(row["perturbed_input_value"].encode()).hexdigest()
        assert replay[key]["coordinates_sha256"] == digest, (
            f"Coordinates changed after replay {key}"
        )
        assert replay[key]["source"] == row["source"]
        assert replay[key]["public_verified"] == row["witness_verified"]
        if key[0] == "candidate" and row["witness_verified"] == "true":
            assert replay[key]["exact_witness_valid"] == "true", (
                f"Accepted screened witness fails exact replay {key}"
            )
    affine = read_rows(audit / "affine-layers.csv")
    assert [int(row["stage"]) for row in affine] == list(
        range(1, {"WK17a": 4, "RSL18a": 2}[network] + 1)
    )
    assert all(
        row["network"] == network and row["native_matches"] == "true" for row in affine
    )
    fallback = {
        image
        for (method, _, image), row in observations.items()
        if method == "candidate" and row["source"] == "MIP"
    }
    certified = {
        image
        for (method, _, image), row in observations.items()
        if method == "candidate" and row["source"] == "backward_bound"
    }
    certificate_sample = (
        certified if network == "RSL18a" else set(sorted(certified)[:10])
    )
    proof_files = [audit / "full-domain-fallback.csv"]
    proof_files.append(
        audit
        / (
            "full-domain-certificates.csv"
            if network == "RSL18a"
            else "full-domain-certificates-first-10.csv"
        )
    )
    proof = {}
    for path in proof_files:
        for row in read_rows(path):
            image = int(row["sample_index"])
            assert image not in proof, f"Duplicate full-domain proof {image}"
            assert row["network"] == network and row["full_domain_enclosed"] == "true"
            proof[image] = row
    assert set(proof) == fallback | certificate_sample, (
        "Bound proofs do not match the selected final cohort"
    )
    assert all(proof[image]["source"] == "MIP" for image in fallback)
    for image in certificate_sample:
        assert proof[image]["source"] == "backward_bound"
        assert proof[image]["negative_target_bounds"] == "9"
    result = {
        "network": network,
        "timed_observations": len(observations),
        "opposing_definitive_outcomes": contradictory,
        "saved_points_replayed": len(replay),
        "fallback_images_with_full_domain_proofs": len(fallback),
        "directly_certified_images": len(certified),
        "direct_certificate_images_with_full_domain_proofs": sorted(certificate_sample),
        "hidden_intervals_checked": sum(
            int(row["hidden_intervals_checked"]) for row in proof.values()
        ),
        "target_bounds_checked": sum(
            int(row["target_bounds_checked"]) for row in proof.values()
        ),
        "native_affine_coefficients_checked": sum(
            int(row["coefficients_checked"]) for row in affine
        ),
        "native_affine_biases_checked": sum(
            int(row["biases_checked"]) for row in affine
        ),
        "methods": {
            method: {
                "observations": sum(key[0] == method for key in observations),
                "saved_points": sum(key[0] == method for key in replay),
                "original_input_points": sum(
                    key[0] == method and row["source"] == "original_input"
                    for key, row in replay.items()
                ),
                "publicly_verified_points": sum(
                    key[0] == method and row["public_verified"] == "true"
                    for key, row in replay.items()
                ),
                "exact_valid_points": sum(
                    key[0] == method and row["exact_witness_valid"] == "true"
                    for key, row in replay.items()
                ),
            }
            for method in ("baseline", "candidate")
        },
    }
    (audit / "coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    check(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
