# Paired benchmark report

| run | adversarial-example objective |
|---|---|
| master 4f1ed43 [feasibility] | `feasibility` |
| index-key eadb415 [feasibility] | `feasibility` |

Paired per-sample analysis: **index-key eadb415 [feasibility]** vs **master 4f1ed43 [feasibility]**

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 0.37 | 0.69 | 0.79 | 0.90 | 0.99 | 1.18 | 2.44 | 76% | 23% |
| Main solve time | 492 | 0.15 | 0.81 | 0.92 | 1.00 | 1.08 | 1.34 | 4.30 | 45% | 47% |
| Total end-to-end time | 492 | 0.27 | 0.68 | 0.79 | 0.91 | 1.00 | 1.23 | 2.60 | 73% | 24% |
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
| Build + bound tightening | 656 s | 596 s | +61 s | 0.91 | 11% |
| Main solve time | 318 s | 305 s | +13 s | 0.96 | 92% |
| Total end-to-end time | 974 s | 901 s | +74 s | 0.92 | 53% |
| Bound solver calls | 99067 calls | 99067 calls | +0 calls | 1.00 | 0% |

### Solve status (all samples)

| status | master 4f1ed43 [feasibility] | index-key eadb415 [feasibility] |
|---|--:|--:|
| INFEASIBLE | 476 | 476 |
| OPTIMAL | 15 | 16 |
| SKIPPED_PREDICTED_IN_TARGETED | 8 | 8 |
| TIME_LIMIT | 1 | 0 |

### Solve-status changes

| transition | n | samples |
|---|--:|---|
| `TIME_LIMIT` → `OPTIMAL` | 1 | 63 |

### Semantic-outcome changes

| transition | n | samples |
|---|--:|---|
| `time_limit_unresolved` → `adversarial_example_found_or_best_known` | 1 | 63 |
| `witness_verification_failed` → `adversarial_example_found_or_best_known` | 1 | 480 |
