# Garbage collection during WK17a screening

The green and blue curves in the [relative-cost plot](../plots/ratio_ecdf.png) each rise twice. In
the original benchmark, many images that backward bounds certified directly had one fast screened
call and one slow call. The benchmark recorded elapsed time for each call. We reran eight of the
same images and measured how much time Julia spent in garbage collection (GC).

| Screened call duration | Calls | Median elapsed | Median GC | Median elapsed minus GC |
| ---------------------- | ----: | -------------: | --------: | ----------------------: |
| Under 0.25 s           |    53 |        0.090 s |   0.009 s |                 0.077 s |
| At least 0.25 s        |    11 |        0.437 s |   0.366 s |                 0.072 s |

Each column uses per-call measurements. The 0.25 s cutoff lies in the gap between the longest fast
call (0.146 s) and the shortest slow call (0.375 s). GC accounted for 81–86% of elapsed time in
**every slow call**. After subtracting GC time, each took 0.069–0.077 s. All 64 calls certified
their image directly, allocated about 120 MiB each, and recorded zero compilation time.

The delay moved between images. Image 5 originally took 0.089 and 0.103 s. In the rerun, one call on
image 5 took 0.520 s, including 0.448 s in GC. Image 113 originally had a 0.568 s call and had eight
fast calls in the rerun. This explains how a second rise can form even when both calls take the same
verifier path. The original 2,000-image run had no per-call GC counters, so the evidence for GC
comes from the rerun.

## Measurements

We used the [original benchmark](../README.md)'s WK17a network, Julia 1.12.6 environment, public API
settings, warm-up images and CPU 0 on the same Intel Core i7-8700K. This probe used one worker. From
the original CPU-0 lane we took the first four directly certified images in each of two groups:

- Images 5, 13, 21 and 29 had screened-to-baseline time ratios from 0.10 to 0.30 and both screened
  calls took under 0.16 s.
- Images 25, 49, 61 and 113 had ratios from 0.55 to 1.00, one screened call under 0.16 s and one
  between 0.50 and 0.70 s.

We ran those images in test-index order for four cycles. Each cycle kept the benchmark's two
screened and two baseline calls per image, in the original order. We timed each public call with
Julia's `@timed` and let GC run normally. The [raw observations](times.csv) include all 128 calls
and six warm-ups (cycle 0). The table uses the 64 screened calls after warm-up. This sample was
selected to study the two timing patterns, so its slow-call frequency should not be extrapolated to
the full cohort.

To reproduce, use the [pinned benchmark environment](../README.md#reproduce) and run the
[measurement script](run.jl) from the repository root on the same core:

```sh
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0 \
  julia --project=benchmarks benchmarks/reports/2026-11-28-public-default/gc-timing/run.jl /tmp/wk17a-gc.csv
```
