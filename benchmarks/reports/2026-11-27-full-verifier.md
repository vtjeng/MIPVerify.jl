# Time the complete verifier with both affine screens and corner checks

We ran the complete verifier on the first 2,000 WK17a and first 100 RSL18a MNIST test images. All
three configurations use adaptive backward bounds and check their maximizing pixel corners before
targeted PGD. The second adds the forward parallelogram screen on remaining rivals. The third also
tries its pixel corners before PGD. Every image starts from its original pixels in every
configuration and repeat. Each fallback builds and solves its own LP/MIP model.

## WK17a on 2,000 images

The forward screen added 18.3 seconds on average (4.6%). Checking its corners then saved 16.5
seconds on average (3.9%). Together, the additions avoided five solver referrals. The full verifier
took 6.0 seconds longer in run 1 and 2.3 seconds less in run 2.

| Full verifier configuration      | Total seconds, runs 1 / 2 | Images entering LP/MIP | Final MIP seconds, runs 1 / 2 | Bound LP calls, runs 1 / 2 |
| -------------------------------- | ------------------------: | ---------------------: | ----------------------------: | -------------------------: |
| Backward screen and first corner |             398.3 / 397.6 |                     34 |                 229.5 / 227.2 |                        104 |
| Add forward parallelograms       |             423.3 / 409.2 |                     32 |                 234.1 / 224.4 |                         98 |
| Also check forward corners       |             404.3 / 395.2 |                     29 |                 221.5 / 213.5 |                         85 |

| Paired change       | Seconds saved, runs 1 / 2 | Percent saved, runs 1 / 2 | Repeat variation | Images faster / slower | Per-image savings, p10 / median / p90 (ms) |
| ------------------- | ------------------------: | ------------------------: | ---------------: | ---------------------: | -----------------------------------------: |
| Forward screen      |             -25.0 / -11.6 |               -6.3 / -2.9 |             3.4% |             944 / 1056 |                         -15.2 / -0.4 / 8.5 |
| Second corner check |               19.0 / 14.0 |                 4.5 / 3.4 |             3.4% |             1018 / 982 |                           -9.0 / 0.1 / 9.7 |
| Both additions      |                -6.0 / 2.3 |                -1.5 / 0.6 |             2.3% |             927 / 1073 |                         -13.6 / -0.6 / 8.6 |

| Paired change       | Image IDs skipping LP/MIP, runs 1 / 2 | Newly referred image IDs, runs 1 / 2 | Prior final MIP seconds on skipped images, runs 1 / 2 | End-to-end seconds saved on those images, runs 1 / 2 |
| ------------------- | ------------------------------------- | ------------------------------------ | ----------------------------------------------------: | ---------------------------------------------------: |
| Forward screen      | 544, 724                              | none                                 |                                             0.1 / 0.1 |                                            0.9 / 0.9 |
| Second corner check | 480, 952, 1193                        | none                                 |                                           11.4 / 11.4 |                                          17.7 / 15.6 |

| Configuration                    | Robust images, runs 1 / 2 | Checked adversarial examples, runs 1 / 2 | Unresolved, runs 1 / 2 |
| -------------------------------- | ------------------------: | ---------------------------------------: | ---------------------: |
| Backward screen and first corner |                      1874 |                                      126 |                      0 |
| Add forward parallelograms       |                      1874 |                                      126 |                      0 |
| Also check forward corners       |                      1874 |                                      126 |                      0 |

Every image received the same verdict across the three configurations within each repeat.

## RSL18a on 100 images

The forward screen added 15.6 seconds on average (0.3%). Checking its corners then saved 20.0
seconds on average (0.4%). Both differences are within the 0.52% variation between repeats of
forward screening alone. Together the additions saved 3.1 and 5.7 seconds on totals exceeding 5,500
seconds. Every configuration sent the same 49 images to the solver. In run 1, forward screening
alone reached the time limit on image 77. The other configurations certified it.

| Full verifier configuration      | Total seconds, runs 1 / 2 | Images entering LP/MIP | Final MIP seconds, runs 1 / 2 | Bound LP calls, runs 1 / 2 |
| -------------------------------- | ------------------------: | ---------------------: | ----------------------------: | -------------------------: |
| Backward screen and first corner |         5,553.0 / 5,551.5 |                     49 |             5,448.2 / 5,446.7 |                        441 |
| Add forward parallelograms       |         5,582.4 / 5,553.3 |                     49 |             5,478.9 / 5,447.5 |                        441 |
| Also check forward corners       |         5,550.0 / 5,545.8 |                     49 |             5,444.7 / 5,443.5 |                        441 |

| Paired change       | Seconds saved, runs 1 / 2 | Percent saved, runs 1 / 2 | Repeat variation | Images faster / slower | Per-image savings, p10 / median / p90 (ms) |
| ------------------- | ------------------------: | ------------------------: | ---------------: | ---------------------: | -----------------------------------------: |
| Forward screen      |              -29.4 / -1.8 |               -0.5 / -0.0 |             0.5% |                49 / 51 |                      -277.1 / -0.0 / 363.9 |
| Second corner check |                32.5 / 7.5 |                 0.6 / 0.1 |             0.5% |                49 / 51 |                      -281.7 / -0.1 / 356.4 |
| Both additions      |                 3.1 / 5.7 |                 0.1 / 0.1 |             0.1% |                48 / 52 |                      -163.0 / -0.0 / 452.4 |

| Paired change       | Image IDs skipping LP/MIP, runs 1 / 2 | Newly referred image IDs, runs 1 / 2 | Prior final MIP seconds on skipped images, runs 1 / 2 | End-to-end seconds saved on those images, runs 1 / 2 |
| ------------------- | ------------------------------------- | ------------------------------------ | ----------------------------------------------------: | ---------------------------------------------------: |
| Forward screen      | none                                  | none                                 |                                             0.0 / 0.0 |                                            0.0 / 0.0 |
| Second corner check | none                                  | none                                 |                                             0.0 / 0.0 |                                            0.0 / 0.0 |

| Configuration                    | Robust images, runs 1 / 2 | Checked adversarial examples, runs 1 / 2 | Unresolved, runs 1 / 2 |
| -------------------------------- | ------------------------: | ---------------------------------------: | ---------------------: |
| Backward screen and first corner |                        42 |                                       16 |                     42 |
| Add forward parallelograms       |                   41 / 42 |                                       16 |                43 / 42 |
| Also check forward corners       |                        42 |                                       16 |                     42 |

## Measurement

The implementation is the verifier from `0348085` with two switches for this ablation. Five Julia
1.12.6 processes each used one physical core of an Intel Core i7-8700K and single-threaded BLAS and
HiGHS 1.14.0. Every image used a pixel budget of 0.10, a 20-second limit per target LP and a
120-second limit for the final MIP. The configurations rotated by image and ran in reverse order on
the repeat. Each image used the same worker core in all six runs. The batch stopped after 12,571 of
12,600 measurements. We completed the 29 missing RSL18a measurements on their original core after
repeating the worker warm-up.

A time total sums per-image elapsed times from entering the verifier until its verdict or timeout.
It includes bound construction, candidate validation, PGD, model construction, all LPs and the final
MIP. It excludes loading the model and data, worker warm-up and writing the results. The final MIP
column times the final optimizer call alone. Five workers processed different images concurrently,
so the sum differs from elapsed time for the whole batch.

Positive savings mean the added step reduced total time. The reported repeat variation is the larger
of the two configurations’ absolute run-to-run differences divided by their respective means. It
describes repeatability on this fixed cohort. We average the two differences per image for the
faster/slower counts and quantiles.

[Per-image times, outcomes, LP calls and witness sources](2026-11-27-full-verifier.csv.gz) ·
[Benchmark runner](end_to_end_screens.jl)

```sh
for worker in 1 2 3 4 5; do
  cpu=$((2 * (worker - 1)))
  OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 taskset -c "$cpu" julia --project=benchmarks \
    benchmarks/reports/end_to_end_screens.jl "worker-$worker.csv" "$worker" 5 &
done
wait
```
