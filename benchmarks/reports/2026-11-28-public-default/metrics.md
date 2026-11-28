### Timing totals

| Phase                 | Target LP (s) | Backward bounds (s) | Seconds saved | Reduction |
| --------------------- | ------------: | ------------------: | ------------: | --------: |
| Outside final solve   |         910.9 |               330.7 |         580.3 |     63.7% |
| Final MIP solve       |         776.0 |               220.2 |         555.8 |     71.6% |
| Total end-to-end time |       1,687.0 |               550.9 |       1,136.1 |     67.3% |

Totals sum each image's mean across its two observations.

### Per-image ratios

| Measure               | Eligible pairs | 10th percentile | Median | 90th percentile | Faster by >1% | Within 1% | Slower by >1% |
| --------------------- | -------------: | --------------: | -----: | --------------: | ------------: | --------: | ------------: |
| Outside final solve   |           2000 |           0.176 |  0.225 |           0.824 |          1930 |         7 |            63 |
| Final MIP solve       |           1951 |           0.000 |  0.000 |           0.000 |          1941 |         0 |            10 |
| Total end-to-end time |           2000 |           0.156 |  0.206 |           0.750 |          1949 |         4 |            47 |
| Bound solver calls    |            777 |           0.000 |  0.000 |           0.000 |           761 |        15 |             1 |

Ratios are Backward bounds divided by Target LP. A zero Target LP value has no finite ratio and is
omitted only from ratio statistics. Absolute plots retain all images and true zeros.

The 95% paired-image bootstrap interval for the total-time reduction is 58.2–75.6%. It measures
sensitivity to the image mix.

Removing the ten images with the largest time savings leaves a 65.7% reduction.

Target LP complete-run totals were 1,690.2 s, 1,683.8 s.

Backward bounds complete-run totals were 554.7 s, 547.1 s.

### First 500 images requiring the integer solver in every observation

The intersection requires source `MIP` and an actual solver-run status in both observations of both
methods. Each image contributes its mean time across those two observations.

| Images | Target LP (s) | Backward bounds (s) | Reduction | Median Backward bounds / Target LP ratio |
| ------ | ------------: | ------------------: | --------: | ---------------------------------------: |
| 7      |        28.313 |              63.957 |   -125.9% |                                    1.501 |

Image IDs: 66, 212, 246, 327, 407, 444, 480.
