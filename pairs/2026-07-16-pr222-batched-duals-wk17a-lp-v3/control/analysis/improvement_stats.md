Paired per-sample analysis: **master control B 8a455e2** vs **master control A 8a455e2**

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 3 | 1.01 | 1.02 | 1.03 | 1.06 | 1.10 | 1.13 | 1.14 | 0% | 67% |
| Main solve time | 3 | 0.83 | 0.85 | 0.89 | 0.94 | 1.04 | 1.10 | 1.14 | 67% | 33% |
| Total end-to-end time | 3 | 0.84 | 0.86 | 0.89 | 0.94 | 1.02 | 1.06 | 1.09 | 67% | 33% |
| Bound solver calls | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0% | 0% |

- `ratio` = candidate ÷ baseline; < 1 = candidate faster. `improved`/`regressed` use a ±1% band.
- `build` = constructing the MIP model; `tightening` = the LP bound-tightening pass; `main solve` = the final verification MIP.
- `total` = `build` + `tightening` + `main solve`.

### Aggregate saving and concentration

- `net saved` = baseline − candidate total; positive = candidate cheaper.
- `pooled ratio` = candidate total ÷ baseline total (aggregate counterpart to the per-sample median).
- `top-10 concentration` = the 10 samples with the largest absolute change account for this share of the total absolute per-sample change (0–100%; higher = a few samples dominate).

| series | baseline | candidate | net saved | pooled ratio | top-10 concentration |
|---|--:|--:|--:|--:|--:|
| Build + bound tightening | 24 s | 24 s | -0 s | 1.02 | 100% |
| Main solve time | 167 s | 156 s | +11 s | 0.93 | 100% |
| Total end-to-end time | 191 s | 180 s | +11 s | 0.94 | 100% |
| Bound solver calls | 564 calls | 564 calls | +0 calls | 1.00 | nan% |

### Solve status (all samples)

| status | master control A 8a455e2 | master control B 8a455e2 |
|---|--:|--:|
| OPTIMAL | 3 | 3 |

### Verdict flips — solve status

_None._

### Verdict flips — semantic outcome

_None._
