Paired per-sample analysis: **PR #222 5b40704** vs **post-#209 master 8a455e2**

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 0.55 | 0.72 | 0.77 | 0.81 | 0.85 | 0.94 | 1.28 | 93% | 6% |
| Main solve time | 492 | 0.03 | 0.79 | 0.91 | 1.01 | 1.12 | 1.28 | 5.19 | 42% | 49% |
| Total end-to-end time | 492 | 0.04 | 0.72 | 0.78 | 0.82 | 0.86 | 0.99 | 4.58 | 90% | 8% |
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
| Build + bound tightening | 883 s | 719 s | +164 s | 0.81 | 6% |
| Main solve time | 1549 s | 1552 s | -3 s | 1.00 | 91% |
| Total end-to-end time | 2432 s | 2271 s | +161 s | 0.93 | 65% |
| Bound solver calls | 99067 calls | 99067 calls | +0 calls | 1.00 | nan% |

### Solve status (all samples)

| status | post-#209 master 8a455e2 | PR #222 5b40704 |
|---|--:|--:|
| INFEASIBLE | 475 | 476 |
| OPTIMAL | 10 | 10 |
| SKIPPED_PREDICTED_IN_TARGETED | 8 | 8 |
| TIME_LIMIT | 7 | 6 |

### Verdict flips — solve status

| transition | n | samples |
|---|--:|---|
| `OPTIMAL` → `TIME_LIMIT` | 1 | 150 |
| `TIME_LIMIT` → `INFEASIBLE` | 1 | 407 |
| `TIME_LIMIT` → `OPTIMAL` | 1 | 321 |

### Verdict flips — semantic outcome

| transition | n | samples |
|---|--:|---|
| `time_limit_unresolved` → `adversarial_example_found_or_best_known` | 1 | 212 |
| `time_limit_unresolved` → `certified_no_adversarial_example` | 1 | 407 |
