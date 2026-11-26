### Timing totals

| Phase | Target LP (s) | Backward bounds (s) | Seconds saved | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Outside final solve | 1,034.2 | 221.2 | 813.0 | 78.6% |
| Final MIP solve | 808.3 | 253.3 | 555.0 | 68.7% |
| Total end-to-end time | 1,842.5 | 474.4 | 1,368.1 | 74.3% |

Totals sum each image's mean across its two observations.

### Per-image ratios

| Measure | Eligible pairs | 10th percentile | Median | 90th percentile | Faster by >1% | Within 1% | Slower by >1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Outside final solve | 2000 | 0.106 | 0.154 | 0.201 | 1926 | 1 | 73 |
| Final MIP solve | 1951 | 0.000 | 0.000 | 0.000 | 1942 | 2 | 7 |
| Total end-to-end time | 2000 | 0.093 | 0.140 | 0.184 | 1933 | 1 | 66 |
| Bound solver calls | 777 | 0.000 | 0.000 | 0.000 | 761 | 15 | 1 |

Ratios are Backward bounds divided by Target LP. A zero Target LP value has no finite ratio and is omitted only from ratio statistics. Absolute plots retain all images and true zeros.

The 95% paired-image bootstrap interval for the total-time reduction is 63.0–84.3%. It measures sensitivity to the image mix.

Removing the ten images with the largest time savings leaves a 76.0% reduction.

Target LP complete-run totals were 1,812.0 s, 1,873.0 s.

Backward bounds complete-run totals were 445.5 s, 503.4 s.

### First 500 images requiring the integer solver in every observation

The intersection requires source `MIP` and an actual solver-run status in both observations of both methods. Each image contributes its mean time across those two observations.

| Images | Target LP (s) | Backward bounds (s) | Reduction | Median Backward bounds / Target LP ratio |
| --- | ---: | ---: | ---: | ---: |
| 7 | 28.971 | 104.771 | -261.6% | 1.584 |

Image IDs: 66, 212, 246, 327, 407, 444, 480.
