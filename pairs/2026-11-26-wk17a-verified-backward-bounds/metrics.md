### Timing totals

| Phase | Target LP (s) | Backward bounds (s) | Seconds saved | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Outside final solve | 990.5 | 225.6 | 764.9 | 77.2% |
| Final MIP solve | 781.3 | 215.6 | 565.7 | 72.4% |
| Total end-to-end time | 1,771.9 | 441.3 | 1,330.6 | 75.1% |

Totals sum each image's mean across its two observations.

### Per-image ratios

| Measure | Eligible pairs | 10th percentile | Median | 90th percentile | Faster by >1% | Within 1% | Slower by >1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Outside final solve | 2000 | 0.112 | 0.155 | 0.202 | 1926 | 3 | 71 |
| Final MIP solve | 1951 | 0.000 | 0.000 | 0.000 | 1941 | 0 | 10 |
| Total end-to-end time | 2000 | 0.098 | 0.141 | 0.184 | 1935 | 1 | 64 |
| Bound solver calls | 777 | 0.000 | 0.000 | 0.000 | 761 | 15 | 1 |

Ratios are Backward bounds divided by Target LP. A zero Target LP value has no finite ratio and is omitted only from ratio statistics. Absolute plots retain all images and true zeros.

The 95% paired-image bootstrap interval for the total-time reduction is 65.4–84.1%. It measures sensitivity to the image mix.

Removing the ten images with the largest time savings leaves a 75.8% reduction.

Target LP complete-run totals were 1,790.8 s, 1,753.0 s.

Backward bounds complete-run totals were 464.1 s, 418.5 s.

### First 500 images requiring the integer solver in every observation

The intersection requires source `MIP` and an actual solver-run status in both observations of both methods. Each image contributes its mean time across those two observations.

| Images | Target LP (s) | Backward bounds (s) | Reduction | Median Backward bounds / Target LP ratio |
| --- | ---: | ---: | ---: | ---: |
| 7 | 27.646 | 93.446 | -238.0% | 1.541 |

Image IDs: 66, 212, 246, 327, 407, 444, 480.
