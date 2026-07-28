# Paired benchmark report

| run | adversarial-example objective |
|---|---|
| master 4f1ed43 [feasibility] | `feasibility` |
| moi-indices 9c12298 [feasibility] | `feasibility` |

Paired per-sample analysis: **moi-indices 9c12298 [feasibility]** vs **master 4f1ed43 [feasibility]**

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 0.40 | 0.77 | 0.84 | 0.88 | 1.02 | 1.31 | 2.46 | 74% | 25% |
| Main solve time | 492 | 0.10 | 0.92 | 0.97 | 1.02 | 1.10 | 1.38 | 3.80 | 34% | 55% |
| Total end-to-end time | 492 | 0.11 | 0.76 | 0.84 | 0.89 | 1.04 | 1.32 | 3.12 | 73% | 27% |
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
| Build + bound tightening | 608 s | 584 s | +24 s | 0.96 | 12% |
| Main solve time | 440 s | 374 s | +66 s | 0.85 | 94% |
| Total end-to-end time | 1048 s | 958 s | +90 s | 0.91 | 62% |
| Bound solver calls | 99067 calls | 99067 calls | +0 calls | 1.00 | 0% |

### Solve status (all samples)

| status | master 4f1ed43 [feasibility] | moi-indices 9c12298 [feasibility] |
|---|--:|--:|
| INFEASIBLE | 475 | 476 |
| OPTIMAL | 15 | 15 |
| SKIPPED_PREDICTED_IN_TARGETED | 8 | 8 |
| TIME_LIMIT | 2 | 1 |

### Solve-status changes

| transition | n | samples |
|---|--:|---|
| `TIME_LIMIT` → `INFEASIBLE` | 1 | 19 |

### Semantic-outcome changes

| transition | n | samples |
|---|--:|---|
| `time_limit_unresolved` → `certified_no_adversarial_example` | 1 | 19 |
| `witness_verification_failed` → `adversarial_example_found_or_best_known` | 1 | 212 |
