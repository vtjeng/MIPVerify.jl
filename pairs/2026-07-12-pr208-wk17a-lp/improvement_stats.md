Paired per-sample analysis: **PR#208 f8e02c2** vs **master 1bb2f9d**  
Ratio = candidate / baseline; < 1 is faster/fewer. Improved/regressed use a ±1% band.  
**Total end-to-end = Build + bound tightening + Main solve** (build + bound tightening is defined as total − main solve, so the two components sum to the total).

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 1.09 | 1.30 | 1.39 | 1.47 | 1.56 | 1.65 | 2.61 | 0% | 100% |
| Main solve time | 492 | 0.32 | 0.91 | 0.96 | 1.01 | 1.06 | 1.14 | 2.40 | 40% | 48% |
| Total end-to-end time | 492 | 0.52 | 1.28 | 1.37 | 1.46 | 1.56 | 1.66 | 2.60 | 1% | 99% |

### Aggregate saving and concentration

Net saved = baseline − candidate total (positive = candidate cheaper). Pooled = candidate ÷ baseline total, the aggregate counterpart to the per-sample median. Concentration = share of the total absolute per-sample change from the 10 biggest movers.

| series | baseline | candidate | net saved | pooled | top-10 concentration |
|---|--:|--:|--:|--:|--:|
| Build + bound tightening | 2334 s | 3383 s | -1049 s | 1.45 | 4% |
| Main solve time | 1438 s | 1513 s | -74 s | 1.05 | 89% |
| Total end-to-end time | 3772 s | 4895 s | -1123 s | 1.30 | 25% |

### Solve status (all samples)

| status | master 1bb2f9d | PR#208 f8e02c2 |
|---|--:|--:|
| INFEASIBLE | 475 | 476 |
| OPTIMAL | 11 | 9 |
| SKIPPED_PREDICTED_IN_TARGETED | 8 | 8 |
| TIME_LIMIT | 6 | 7 |

### Verdict flips (7)

- **sample 150**: status `TIME_LIMIT` → `OPTIMAL`
- **sample 242**: status `OPTIMAL` → `TIME_LIMIT`
- **sample 246**: status `TIME_LIMIT` → `INFEASIBLE`; outcome `time_limit_unresolved` → `certified_no_adversarial_example`
- **sample 445**: status `OPTIMAL` → `TIME_LIMIT`
- **sample 446**: status `OPTIMAL` → `TIME_LIMIT`
- **sample 449**: status `TIME_LIMIT` → `OPTIMAL`
- **sample 496**: status `OPTIMAL` → `TIME_LIMIT`
