# Performance log

Measured performance changes, one row per PR, newest first. Append a row in the same PR that changes
performance, with the measured impact and an evidence link.

## Verification (solve) performance

Changes to solve and tightening work. These scale with the number of samples verified.

| Date       | PR   | Change                                                         | Measured impact                                                                                                                                                                            | Evidence                                                                                                                                                                                      |
| ---------- | ---- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-07-16 | #209 | Screen ReLU bounds progressively (interval, then LP, then MIP) | Pooled solver wall clock −59% on the 500-sample WK17a LP benchmark; median per-sample 4.3× faster; main solve unchanged. First post-merge nightly (2026-07-17) pending confirmation.       | [Report](https://github.com/vtjeng/MIPVerify.jl/pull/209#issuecomment-4989944753), [pair data](https://github.com/vtjeng/MIPVerify.jl/tree/benchmark-reports/pairs/2026-07-15-pr209-wk17a-lp) |
| 2026-07-13 | #208 | Certify LP tightening bounds from row duals                    | Pooled solver wall clock +~30% (soundness fix: stops trusting solver-reported objectives). Nightly 500-sample run rose from 52–63 to 78–82 min. #209 pays this back: net −40% vs pre-#208. | [Retroactive report](https://github.com/vtjeng/MIPVerify.jl/pull/208#issuecomment-4988727305)                                                                                                 |

## CI-only improvements

Fixed per-run cost of the test and benchmark workflows. These do not scale with samples.

| Date       | PR   | Change                                            | Measured impact                                                                 | Evidence                                                                                                                                                  |
| ---------- | ---- | ------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-07-16 | #237 | macOS test legs on native aarch64 Julia           | macOS legs 719–729s → 352–355s each                                             | Runs [before](https://github.com/vtjeng/MIPVerify.jl/actions/runs/29483424425) / [after](https://github.com/vtjeng/MIPVerify.jl/actions/runs/29536927187) |
| 2026-07-16 | #235 | actionlint check for workflow files               | +7s per run                                                                     | [After run](https://github.com/vtjeng/MIPVerify.jl/actions/runs/29536927187)                                                                              |
| 2026-07-16 | #230 | Cache Julia package state for docs and benchmarks | Documentation 241s → 148s; Benchmark Helper Tests 208s → 151s                   | Same before/after runs as #237                                                                                                                            |
| 2026-07-16 | #228 | Run notebook checks in one job                    | Four jobs totaling 1790s → one 477s job                                         | Same before/after runs as #237                                                                                                                            |
| 2026-07-16 | #227 | Collect coverage on one CI leg                    | −5m20s aggregate across the six test legs; critical package job 12m09s → 10m11s | [Paired timing in PR](https://github.com/vtjeng/MIPVerify.jl/pull/227)                                                                                    |
| 2026-07-16 | #221 | Reduce default test suite runtime                 | Default suite 742s → 199s (−73%) on a warm local run                            | [Measured impact in PR](https://github.com/vtjeng/MIPVerify.jl/pull/221)                                                                                  |

Aggregate for the 2026-07-16 CI queue (#227–#237): one full CI run fell from 5619s to 3304s of
runner time (−41%), and the longest job from 729s to 503s (−31%).
