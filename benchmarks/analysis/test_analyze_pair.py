import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from analyze_pair import (
    Series,
    build_series_frame,
    load_tightening_algorithm,
    paired_tightening_algorithm,
    series_stats,
    stats_markdown,
)


def _series(key: str, label: str, base_column: str, candidate_column: str) -> Series:
    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_index": frame["sample_index"],
                "base": frame[base_column],
                "cand": frame[candidate_column],
            }
        )

    return Series(key, label, "s", builder)


class SeriesStatsTests(unittest.TestCase):
    def test_zero_movement_has_zero_concentration(self):
        # Both paired values are positive and identical so the total absolute movement is exactly 0.
        merged = pd.DataFrame({"sample_index": [1, 2], "base": [1.0, 2.0], "candidate": [1.0, 2.0]})
        series = _series("identical", "Identical values", "base", "candidate")
        frame = build_series_frame(merged, series)
        stats = series_stats(series, frame)

        self.assertEqual(stats["top10_share"], 0.0)
        self.assertEqual(stats["top5pct_share"], 0.0)

    def test_canceling_changes_keep_nonzero_concentration(self):
        # The +0.5 saving and -0.5 regression cancel in aggregate but still create movement.
        merged = pd.DataFrame({"sample_index": [1, 2], "base": [1.0, 2.0], "candidate": [0.5, 2.5]})
        series = _series("canceling", "Canceling changes", "base", "candidate")
        frame = build_series_frame(merged, series)
        stats = series_stats(series, frame)

        self.assertEqual(stats["net_saved"], 0.0)
        self.assertEqual(stats["top10_share"], 1.0)

    def test_series_can_have_different_eligible_sample_counts(self):
        # The sparse series uses one zero baseline and one missing candidate to exercise both
        # per-series exclusion paths while the full series retains all three samples.
        merged = pd.DataFrame(
            {
                "sample_index": [1, 2, 3],
                "full_base": [1.0, 2.0, 3.0],
                "full_candidate": [1.0, 2.0, 3.0],
                "sparse_base": [1.0, 0.0, 3.0],
                "sparse_candidate": [1.0, 2.0, None],
            }
        )
        full = _series("full", "Full series", "full_base", "full_candidate")
        sparse = _series("sparse", "Sparse series", "sparse_base", "sparse_candidate")

        full_stats = series_stats(full, build_series_frame(merged, full))
        sparse_stats = series_stats(sparse, build_series_frame(merged, sparse))

        self.assertEqual(full_stats["n"], 3)
        self.assertEqual(sparse_stats["n"], 1)


class TighteningMetadataTests(unittest.TestCase):
    def test_stats_footnote_names_each_tightening_mode(self):
        expected_labels = {
            "interval_arithmetic": "interval-arithmetic",
            "lp": "LP",
            "mip": "MIP",
        }
        for algorithm, label in expected_labels.items():
            with self.subTest(algorithm=algorithm):
                markdown = stats_markdown(
                    [], "baseline [closest]", "candidate [closest]", "closest", "closest", algorithm
                )
                self.assertIn(f"`tightening` = the {label} bound-tightening pass", markdown)
                if algorithm != "lp":
                    self.assertNotIn("the LP bound-tightening pass", markdown)

    def test_loads_supported_tightening_modes_from_run_metadata(self):
        for algorithm in ("interval_arithmetic", "lp", "mip"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as run_dir:
                metrics_path = Path(run_dir) / "benchmark_metrics.csv"
                pd.DataFrame({"tightening_algorithm": [algorithm]}).to_csv(
                    metrics_path, index=False
                )

                self.assertEqual(load_tightening_algorithm(Path(run_dir)), algorithm)

    def test_rejects_missing_or_unsupported_tightening_metadata(self):
        with TemporaryDirectory() as run_dir:
            with self.assertRaisesRegex(ValueError, "missing benchmark metadata"):
                load_tightening_algorithm(Path(run_dir))

            metrics_path = Path(run_dir) / "benchmark_metrics.csv"
            pd.DataFrame({"other_column": ["lp"]}).to_csv(metrics_path, index=False)
            with self.assertRaisesRegex(ValueError, "missing tightening_algorithm column"):
                load_tightening_algorithm(Path(run_dir))

            pd.DataFrame({"tightening_algorithm": ["unknown"]}).to_csv(metrics_path, index=False)
            with self.assertRaisesRegex(ValueError, "unsupported tightening_algorithm"):
                load_tightening_algorithm(Path(run_dir))

    def test_rejects_mismatched_paired_tightening_modes(self):
        with TemporaryDirectory() as root_dir:
            baseline_dir = Path(root_dir) / "base"
            candidate_dir = Path(root_dir) / "candidate"
            baseline_dir.mkdir()
            candidate_dir.mkdir()
            pd.DataFrame({"tightening_algorithm": ["lp"]}).to_csv(
                baseline_dir / "benchmark_metrics.csv", index=False
            )
            pd.DataFrame({"tightening_algorithm": ["mip"]}).to_csv(
                candidate_dir / "benchmark_metrics.csv", index=False
            )

            with self.assertRaisesRegex(ValueError, "different tightening algorithms"):
                paired_tightening_algorithm(baseline_dir, candidate_dir)

    def test_cli_writes_mode_aware_footnote(self):
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            baseline_dir = root / "base"
            candidate_dir = root / "candidate"
            output_dir = root / "analysis"
            baseline_dir.mkdir()
            candidate_dir.mkdir()

            # One solved sample with positive build, solve, and total times exercises every required
            # runtime series while keeping the CLI fixture small.
            baseline_sample = pd.DataFrame(
                {
                    "sample_index": [1],
                    "solve_status": ["OPTIMAL"],
                    "total_time_seconds": [2.0],
                    "solve_time_seconds": [1.0],
                }
            )
            candidate_sample = pd.DataFrame(
                {
                    "sample_index": [1],
                    "solve_status": ["OPTIMAL"],
                    "total_time_seconds": [1.5],
                    "solve_time_seconds": [0.75],
                }
            )
            for run_dir, sample in (
                (baseline_dir, baseline_sample),
                (candidate_dir, candidate_sample),
            ):
                sample.to_csv(run_dir / "benchmark_per_sample.csv", index=False)
                pd.DataFrame({"tightening_algorithm": ["mip"]}).to_csv(
                    run_dir / "benchmark_metrics.csv", index=False
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("analyze_pair.py")),
                    "--baseline",
                    str(baseline_dir),
                    "--candidate",
                    str(candidate_dir),
                    "--out",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = (output_dir / "improvement_stats.md").read_text()
            self.assertIn("`tightening` = the MIP bound-tightening pass", report)
            self.assertNotIn("the LP bound-tightening pass", report)


if __name__ == "__main__":
    unittest.main()
