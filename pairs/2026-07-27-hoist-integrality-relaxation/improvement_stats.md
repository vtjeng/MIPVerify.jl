# Paired benchmark report

| run | adversarial-example objective |
|---|---|
| base master [feasibility] | `feasibility` |
| candidate becd8a7 [feasibility] | `feasibility` |

Paired per-sample analysis: **candidate becd8a7 [feasibility]** vs **base master [feasibility]**

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 0.45 | 0.70 | 0.77 | 0.82 | 0.89 | 0.98 | 1.42 | 91% | 9% |
| Main solve time | 492 | 0.14 | 0.85 | 0.94 | 0.99 | 1.04 | 1.14 | 23.23 | 50% | 39% |
| Total end-to-end time | 492 | 0.15 | 0.70 | 0.77 | 0.83 | 0.90 | 1.01 | 8.86 | 89% | 10% |
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
| Build + bound tightening | 602 s | 501 s | +102 s | 0.83 | 8% |
| Main solve time | 451 s | 362 s | +89 s | 0.80 | 96% |
| Total end-to-end time | 1053 s | 862 s | +191 s | 0.82 | 58% |
| Bound solver calls | 99067 calls | 99067 calls | +0 calls | 1.00 | 0% |

### Solve status (all samples)

| status | base master [feasibility] | candidate becd8a7 [feasibility] |
|---|--:|--:|
| INFEASIBLE | 476 | 476 |
| OPTIMAL | 15 | 15 |
| SKIPPED_PREDICTED_IN_TARGETED | 8 | 8 |
| TIME_LIMIT | 1 | 1 |

### Solve-status changes

_None._

### Semantic-outcome changes

_None._
