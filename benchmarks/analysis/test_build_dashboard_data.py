import tempfile
import unittest
from pathlib import Path

from build_dashboard_data import daily_rows


class DashboardArchiveTests(unittest.TestCase):
    def test_legacy_overwrite_and_same_day_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)
            # The legacy rows share one file. Only the middle row matches its
            # runtime, so taking either end of the tracking table gives the wrong median.
            tracking = archive / "tracking.csv"
            tracking.write_text(
                "date,run_id,commit_sha,wall_clock_seconds,median_solve_time_seconds\n"
                "2026-03-06,,first,20,4\n"
                "2026-03-06,,second,10,2\n"
                "2026-03-06,,third,30,6\n"
            )
            day = archive / "runs" / "2026-03-06"
            day.mkdir(parents=True)
            (day / "benchmark_metrics.csv").write_text("wall_clock_seconds,num_samples\n10,1\n")
            (legacy,) = daily_rows(archive)
            self.assertEqual(legacy["commit_sha"], "second")
            self.assertEqual(legacy["median_solve_time_seconds"], 2)
            self.assertNotIn("num_witness_verified", legacy)
            self.assertNotIn("adversarial_example_objective", legacy)

            # Two immutable runs share a date. Append them out of order so the
            # latest timestamp decides which run and which zero count are shown.
            # Modern runs explicitly name their objective in the metrics CSV.
            for hour, verified in [("18", 0), ("06", 1)]:
                run_id = f"2026-03-07T{hour}-00-00Z-revision"
                with tracking.open("a") as file:
                    file.write(f"2026-03-07,{run_id},revision,12,3\n")
                run = archive / "runs" / "2026-03-07" / run_id
                run.mkdir(parents=True)
                (run / "benchmark_metrics.csv").write_text(
                    "wall_clock_seconds,num_witness_verified,adversarial_example_objective,"
                    "backward_screen,num_screen_certified,num_screen_witness\n"
                    f"12,{verified},feasibility,true,1,0\n"
                )
            old, latest = daily_rows(archive)
            self.assertEqual(old, legacy)
            self.assertIn("T18-", latest["run_id"])
            self.assertEqual(latest["num_witness_verified"], 0)
            self.assertEqual(latest.get("adversarial_example_objective"), "feasibility")
            self.assertEqual(latest["backward_screen"], "true")
            self.assertEqual(latest["num_screen_certified"], 1)
            self.assertEqual(latest["num_screen_witness"], 0)
            self.assertNotIn("backward_screen", legacy)
