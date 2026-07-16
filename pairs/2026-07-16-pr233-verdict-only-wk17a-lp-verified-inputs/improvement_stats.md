Paired per-sample analysis: **verdict only PR #233** vs **exact distortion 8a455e2**

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 0.58 | 0.81 | 0.88 | 0.95 | 1.05 | 1.19 | 7.47 | 61% | 34% |
| Main solve time | 492 | 0.01 | 0.68 | 0.79 | 0.90 | 1.01 | 1.14 | 8.81 | 73% | 25% |
| Total end-to-end time | 492 | 0.02 | 0.77 | 0.87 | 0.94 | 1.04 | 1.17 | 7.74 | 62% | 32% |
| Bound solver calls | 492 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0% | 0% |

- `ratio` = candidate ÷ baseline; < 1 = candidate faster. `improved`/`regressed` use a ±1% band.
- `build` = constructing the MIP model; `tightening` = the LP bound-tightening pass; `main solve` = the final verification MIP.
- `total` = `build` + `tightening` + `main solve`.

### Aggregate saving and concentration

- `net saved` = baseline − candidate total; positive = candidate cheaper.
- `pooled ratio` = candidate total ÷ baseline total (aggregate counterpart to the per-sample median).
- `top-10 concentration` = the 10 samples with the largest absolute change account for this share of the total absolute per-sample change (0–100%; higher = a few samples dominate).

| series | baseline | candidate | net saved | pooled ratio | top-10 concentration |
|---|--:|--:|--:|--:|--:|
| Build + bound tightening | 883 s | 909 s | -26 s | 1.03 | 33% |
| Main solve time | 1549 s | 466 s | +1083 s | 0.30 | 79% |
| Total end-to-end time | 2432 s | 1375 s | +1057 s | 0.57 | 71% |
| Bound solver calls | 99067 calls | 99067 calls | +0 calls | 1.00 | nan% |

### Solve status (all samples)

| status | exact distortion 8a455e2 | verdict only PR #233 |
|---|--:|--:|
| INFEASIBLE | 475 | 475 |
| OPTIMAL | 10 | 15 |
| SKIPPED_PREDICTED_IN_TARGETED | 8 | 8 |
| TIME_LIMIT | 7 | 2 |

### Verdict flips — solve status

| transition | n | samples |
|---|--:|---|
| `TIME_LIMIT` → `OPTIMAL` | 5 | 212, 242, 321, 446, 480 |
| `INFEASIBLE` → `TIME_LIMIT` | 1 | 19 |
| `TIME_LIMIT` → `INFEASIBLE` | 1 | 407 |

### Verdict flips — semantic outcome

| transition | n | samples |
|---|--:|---|
| `certified_no_adversarial_example` → `time_limit_unresolved` | 1 | 19 |
| `time_limit_unresolved` → `adversarial_example_found_or_best_known` | 1 | 212 |
| `time_limit_unresolved` → `certified_no_adversarial_example` | 1 | 407 |
