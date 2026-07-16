Paired per-sample analysis: **PR#208 f8e02c2** vs **master 1bb2f9d**

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 1.09 | 1.30 | 1.39 | 1.47 | 1.56 | 1.65 | 2.61 | 0% | 100% |
| Main solve time | 492 | 0.32 | 0.91 | 0.96 | 1.01 | 1.06 | 1.14 | 2.40 | 40% | 48% |
| Total end-to-end time | 492 | 0.52 | 1.28 | 1.37 | 1.46 | 1.56 | 1.66 | 2.60 | 1% | 99% |

- `ratio` = candidate ÷ baseline; < 1 = candidate faster. `improved`/`regressed` use a ±1% band.
- `build` = constructing the MIP model; `tightening` = the LP bound-tightening pass; `main solve` = the final verification MIP.
- `total` = `build` + `tightening` + `main solve`.

### Aggregate saving and concentration

- `net saved` = baseline − candidate total; positive = candidate cheaper.
- `pooled ratio` = candidate total ÷ baseline total (aggregate counterpart to the per-sample median).
- `top-10 concentration` = the 10 samples with the largest absolute change account for this share of the total absolute per-sample change (0–100%; higher = a few samples dominate).

| series | baseline | candidate | net saved | pooled ratio | top-10 concentration |
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

### Verdict flips — solve status

| transition | n | samples |
|---|--:|---|
| `OPTIMAL` → `TIME_LIMIT` | 4 | 242, 445, 446, 496 |
| `TIME_LIMIT` → `OPTIMAL` | 2 | 150, 449 |
| `TIME_LIMIT` → `INFEASIBLE` | 1 | 246 |

### Verdict flips — semantic outcome

| transition | n | samples |
|---|--:|---|
| `time_limit_unresolved` → `certified_no_adversarial_example` | 1 | 246 |
