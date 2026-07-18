import unittest

import pandas as pd

from analyze_pair import Series, build_series_frame, series_stats


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


if __name__ == "__main__":
    unittest.main()
