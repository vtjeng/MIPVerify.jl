# Paired benchmark report

| run | adversarial-example objective |
|---|---|
| closest master ee5fcfa [closest] | `closest` |
| feasibility PR #233 edfcbdf [feasibility] | `feasibility` |

> **Cross-objective comparison.** These runs use different solve goals. Interpret the timings as the performance tradeoff between exact distortion and a feasibility solve.

Paired per-sample analysis: **feasibility PR #233 edfcbdf [feasibility]** vs **closest master ee5fcfa [closest]**

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 0.25 | 0.80 | 0.94 | 1.04 | 1.11 | 1.20 | 1.73 | 34% | 62% |
| Main solve time | 492 | 0.01 | 0.65 | 0.85 | 0.95 | 1.07 | 1.23 | 11.37 | 59% | 37% |
| Total end-to-end time | 492 | 0.03 | 0.72 | 0.91 | 1.03 | 1.10 | 1.21 | 9.96 | 38% | 58% |
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
| Build + bound tightening | 689 s | 675 s | +13 s | 0.98 | 22% |
| Main solve time | 1451 s | 622 s | +829 s | 0.43 | 80% |
| Total end-to-end time | 2140 s | 1297 s | +842 s | 0.61 | 74% |
| Bound solver calls | 99067 calls | 99067 calls | +0 calls | 1.00 | nan% |

### Solve status (all samples)

| status | closest master ee5fcfa [closest] | feasibility PR #233 edfcbdf [feasibility] |
|---|--:|--:|
| INFEASIBLE | 476 | 474 |
| OPTIMAL | 10 | 14 |
| SKIPPED_PREDICTED_IN_TARGETED | 8 | 8 |
| TIME_LIMIT | 6 | 4 |

### Solve-status changes

| transition | n | samples |
|---|--:|---|
| `TIME_LIMIT` → `OPTIMAL` | 4 | 150, 242, 446, 480 |
| `INFEASIBLE` → `TIME_LIMIT` | 2 | 19, 46 |

### Semantic-outcome changes

| transition | n | samples |
|---|--:|---|
| `certified_no_adversarial_example` → `time_limit_unresolved` | 2 | 19, 46 |
| `adversarial_example_found_or_best_known` → `time_limit_unresolved` | 1 | 212 |
