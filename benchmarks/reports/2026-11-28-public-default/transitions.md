### Solve status and verdict flips

Counts use the first observation of each image under each method.

| Status                          | Target LP | Backward bounds |
| ------------------------------- | --------: | --------------: |
| `INFEASIBLE`                    |      1873 |            1874 |
| `OPTIMAL`                       |        76 |              77 |
| `SKIPPED_PREDICTED_IN_TARGETED` |        49 |              49 |
| `TIME_LIMIT`                    |         2 |               0 |

**Solve status**

| Transition (Target LP → Backward bounds) | Count | MNIST test image IDs |
| ---------------------------------------- | ----: | -------------------: |
| `TIME_LIMIT` → `INFEASIBLE`              |     1 |                  584 |
| `TIME_LIMIT` → `OPTIMAL`                 |     1 |                   63 |

**Semantic outcome**

| Transition (Target LP → Backward bounds)                            | Count | MNIST test image IDs |
| ------------------------------------------------------------------- | ----: | -------------------: |
| `time_limit_unresolved` → `adversarial_example_found_or_best_known` |     1 |                   63 |
| `time_limit_unresolved` → `certified_no_adversarial_example`        |     1 |                  584 |
