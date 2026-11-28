# Check the forward parallelogram's pixel corner before PGD

This historical experiment used the
[verifier and replay driver at the measured revision](https://github.com/vtjeng/MIPVerify.jl/blob/03480855f689d7ce6edd17fc512124a22a6a0522/benchmarks/reports/second_corner.jl).
Check out that revision to reproduce it.

On WK17a, a second set of corner checks found seven adversarial examples after every backward corner
missed. Four of these had previously required PGD and three had reached the solver. On RSL18a, the
additional checks found zero examples. Both repeats gave identical routes per image.

| Network and search order         | Images entering search | Backward corner | Forward corner | PGD witness | Sent to solver | Search seconds, runs 1 / 2 |
| -------------------------------- | ---------------------: | --------------: | -------------: | ----------: | -------------: | -------------------------: |
| WK17a, backward corner then PGD  |                    154 |             117 |              — |           5 |             32 |                5.10 / 5.10 |
| WK17a, both corners then PGD     |                    154 |             117 |              7 |           1 |             29 |                4.95 / 4.62 |
| RSL18a, backward corner then PGD |                     65 |              14 |              — |           2 |             49 |              21.38 / 20.65 |
| RSL18a, both corners then PGD    |                     65 |              14 |              0 |           2 |             49 |              20.99 / 20.94 |

A positive coefficient in the forward margin picks the upper pixel endpoint. A negative coefficient
picks the lower endpoint. Only the original pixel coordinates enter the candidate image. The
verifier evaluates that image on the original network and checks the pixel budget and logits before
accepting it. Within each pass the verifier tries the rival with the highest remaining bound first.
It skips a forward candidate when that rival proposed the same corner in the first pass.

We replayed every image left open by both screens in the
[previous search benchmark](2026-11-27-corner-search.md), using the same bounds, PGD settings and
witness validation. Both paths were warmed and timed twice per image on one pinned Intel Core
i7-8700K core with single-threaded BLAS. Their order alternated by image and reversed on the repeat.
Times sum search and validation across the cohort. Bound construction, storing the corner directions
and subsequent solver work fall outside the timer.
[Per-image routes and timings](2026-11-27-second-corner.csv.gz).

```sh
taskset -c 4 julia --project=benchmarks benchmarks/reports/second_corner.jl second-corner.csv
```
