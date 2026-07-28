# Paired benchmark report

| run | adversarial-example objective |
|---|---|
| base master [feasibility] | `feasibility` |
| candidate 072a027 [feasibility] | `feasibility` |

Paired per-sample analysis: **candidate 072a027 [feasibility]** vs **base master [feasibility]**

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 0.51 | 0.71 | 0.78 | 0.83 | 0.89 | 0.97 | 1.32 | 91% | 8% |
| Main solve time | 492 | 0.41 | 0.84 | 0.93 | 0.97 | 1.01 | 1.06 | 3.94 | 65% | 24% |
| Total end-to-end time | 492 | 0.51 | 0.72 | 0.79 | 0.84 | 0.89 | 1.00 | 2.91 | 89% | 10% |
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
| Build + bound tightening | 602 s | 502 s | +100 s | 0.83 | 7% |
| Main solve time | 451 s | 514 s | -64 s | 1.14 | 94% |
| Total end-to-end time | 1053 s | 1017 s | +36 s | 0.97 | 50% |
| Bound solver calls | 99067 calls | 99067 calls | +0 calls | 1.00 | 0% |

### Solve status (all samples)

| status | base master [feasibility] | candidate 072a027 [feasibility] |
|---|--:|--:|
| INFEASIBLE | 476 | 476 |
| OPTIMAL | 15 | 14 |
| SKIPPED_PREDICTED_IN_TARGETED | 8 | 8 |
| TIME_LIMIT | 1 | 2 |

### Solve-status changes

| transition | n | samples |
|---|--:|---|
| `OPTIMAL` → `TIME_LIMIT` | 1 | 480 |

### Semantic-outcome changes

| transition | n | samples |
|---|--:|---|
| `adversarial_example_found_or_best_known` → `time_limit_unresolved` | 1 | 480 |
