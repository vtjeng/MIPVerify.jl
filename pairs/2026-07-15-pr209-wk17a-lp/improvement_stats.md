Paired per-sample analysis: **PR#209 6bf062b** vs **base ed9194c**

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 0.02 | 0.17 | 0.20 | 0.23 | 0.26 | 0.31 | 5.44 | 100% | 0% |
| Main solve time | 492 | 0.06 | 0.57 | 0.81 | 1.00 | 1.19 | 1.45 | 17.83 | 48% | 47% |
| Total end-to-end time | 492 | 0.02 | 0.17 | 0.20 | 0.23 | 0.28 | 0.36 | 5.45 | 99% | 1% |
| Bound solver calls | 492 | 0.12 | 0.18 | 0.20 | 0.23 | 0.25 | 0.28 | 0.36 | 100% | 0% |

- `ratio` = candidate ÷ baseline; < 1 = candidate faster. `improved`/`regressed` use a ±1% band.
- `build` = constructing the MIP model; `tightening` = the LP bound-tightening pass; `main solve` = the final verification MIP.
- `total` = `build` + `tightening` + `main solve`.

### Aggregate saving and concentration

- `net saved` = baseline − candidate total; positive = candidate cheaper.
- `pooled ratio` = candidate total ÷ baseline total (aggregate counterpart to the per-sample median).
- `top-10 concentration` = the 10 samples with the largest absolute change account for this share of the total absolute per-sample change (0–100%; higher = a few samples dominate).

| series | baseline | candidate | net saved | pooled ratio | top-10 concentration |
|---|--:|--:|--:|--:|--:|
| Build + bound tightening | 4914 s | 1109 s | +3805 s | 0.23 | 13% |
| Main solve time | 1532 s | 1526 s | +7 s | 1.00 | 86% |
| Total end-to-end time | 6446 s | 2634 s | +3812 s | 0.41 | 14% |
| Bound solver calls | 427076 calls | 99067 calls | +328009 calls | 0.23 | 2% |

### Solve status (all samples)

| status | base ed9194c | PR#209 6bf062b |
|---|--:|--:|
| INFEASIBLE | 476 | 476 |
| OPTIMAL | 10 | 9 |
| SKIPPED_PREDICTED_IN_TARGETED | 8 | 8 |
| TIME_LIMIT | 6 | 7 |

### Verdict flips — solve status

| transition | n | samples |
|---|--:|---|
| `OPTIMAL` → `TIME_LIMIT` | 2 | 150, 449 |
| `TIME_LIMIT` → `OPTIMAL` | 1 | 242 |

### Verdict flips — semantic outcome

| transition | n | samples |
|---|--:|---|
| `time_limit_unresolved` → `adversarial_example_found_or_best_known` | 2 | 212, 446 |
| `adversarial_example_found_or_best_known` → `time_limit_unresolved` | 1 | 321 |
