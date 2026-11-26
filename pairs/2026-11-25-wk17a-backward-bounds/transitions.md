### Solve status and verdict flips

Counts use the first observation of each image under each method.

| Status | Target LP | Backward bounds |
| --- | ---: | ---: |
| `INFEASIBLE` | 1872 | 30 |
| `NO_SOLVER` | 0 | 1966 |
| `OPTIMAL` | 76 | 4 |
| `SKIPPED_PREDICTED_IN_TARGETED` | 49 | 0 |
| `TIME_LIMIT` | 3 | 0 |

`NO_SOLVER` records a result reached without running the integer solver. The verification-outcome table below distinguishes robustness certificates from adversarial examples.

**Solve status**

| Transition (Target LP → Backward bounds) | Count | MNIST test image IDs |
| --- | ---: | ---: |
| `INFEASIBLE` → `NO_SOLVER` | 1844 | [All 1,844 image IDs](image-ids/solve_status-INFEASIBLE-to-NO_SOLVER.csv) |
| `OPTIMAL` → `NO_SOLVER` | 72 | [All 72 image IDs](image-ids/solve_status-OPTIMAL-to-NO_SOLVER.csv) |
| `SKIPPED_PREDICTED_IN_TARGETED` → `NO_SOLVER` | 49 | [All 49 image IDs](image-ids/solve_status-SKIPPED_PREDICTED_IN_TARGETED-to-NO_SOLVER.csv) |
| `TIME_LIMIT` → `INFEASIBLE` | 2 | 584, 1572 |
| `TIME_LIMIT` → `NO_SOLVER` | 1 | 63 |

**Semantic outcome**

| Transition (Target LP → Backward bounds) | Count | MNIST test image IDs |
| --- | ---: | ---: |
| `time_limit_unresolved` → `adversarial_example_found_or_best_known` | 1 | 63 |
| `time_limit_unresolved` → `certified_no_adversarial_example` | 2 | 584, 1572 |
