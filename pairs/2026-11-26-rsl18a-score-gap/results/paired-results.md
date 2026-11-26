# RSL18a paired results

Times average both observations per image. Statuses and outcomes are aligned by image and observation, never by CSV row position.

| Method | Observation | Seconds | Bound calls | Main solve (s) |
| :--- | ---: | ---: | ---: | ---: |
| Target LP | 1 | 7233.356 | 882 | 7071.101 |
| Target LP | 2 | 7256.312 | 882 | 7074.217 |
| Backward score-gap screen | 1 | 5478.073 | 441 | 5376.353 |
| Backward score-gap screen | 2 | 5463.354 | 441 | 5364.766 |

## Status counts

| Method | Observation | Value | Images |
| :--- | ---: | :--- | ---: |
| Target LP | 1 | `INFEASIBLE` | 42 |
| Target LP | 1 | `SKIPPED_PREDICTED_IN_TARGETED` | 2 |
| Target LP | 1 | `TIME_LIMIT` | 56 |
| Target LP | 2 | `INFEASIBLE` | 42 |
| Target LP | 2 | `SKIPPED_PREDICTED_IN_TARGETED` | 2 |
| Target LP | 2 | `TIME_LIMIT` | 56 |
| Backward score-gap screen | 1 | `INFEASIBLE` | 8 |
| Backward score-gap screen | 1 | `NO_SOLVER` | 51 |
| Backward score-gap screen | 1 | `TIME_LIMIT` | 41 |
| Backward score-gap screen | 2 | `INFEASIBLE` | 8 |
| Backward score-gap screen | 2 | `NO_SOLVER` | 51 |
| Backward score-gap screen | 2 | `TIME_LIMIT` | 41 |

## Status changes

Cross-method rows compare Target LP to Backward score-gap screen. Repeat rows compare observation 1 to observation 2 within the named method.

| Comparison | Transition | Count | Image IDs |
| :--- | :--- | ---: | :--- |
| status_observation_1 | `INFEASIBLE -> NO_SOLVER` | 35 | 1, 4, 6, 11, 14, 15, 18, 26, 28, 29, 31, 33, 36, 38, 49, 50, 52, 56, 57, 61, 69, 70, 71, 72, 80, 82, 83, 84, 86, 87, 89, 91, 92, 94, 100 |
| status_observation_1 | `TIME_LIMIT -> NO_SOLVER` | 14 | 2, 9, 19, 34, 39, 44, 60, 62, 63, 64, 65, 88, 93, 98 |
| status_observation_1 | `SKIPPED_PREDICTED_IN_TARGETED -> NO_SOLVER` | 2 | 74, 78 |
| status_observation_1 | `TIME_LIMIT -> INFEASIBLE` | 1 | 77 |
| status_observation_2 | `INFEASIBLE -> NO_SOLVER` | 35 | 1, 4, 6, 11, 14, 15, 18, 26, 28, 29, 31, 33, 36, 38, 49, 50, 52, 56, 57, 61, 69, 70, 71, 72, 80, 82, 83, 84, 86, 87, 89, 91, 92, 94, 100 |
| status_observation_2 | `TIME_LIMIT -> NO_SOLVER` | 14 | 2, 9, 19, 34, 39, 44, 60, 62, 63, 64, 65, 88, 93, 98 |
| status_observation_2 | `SKIPPED_PREDICTED_IN_TARGETED -> NO_SOLVER` | 2 | 74, 78 |
| status_observation_2 | `TIME_LIMIT -> INFEASIBLE` | 1 | 77 |
| status_repeat_target_lp | None | 0 | — |
| status_repeat_screen | None | 0 | — |

## Outcome counts

| Method | Observation | Value | Images |
| :--- | ---: | :--- | ---: |
| Target LP | 1 | `adversarial_example_found_or_best_known` | 2 |
| Target LP | 1 | `certified_no_adversarial_example` | 42 |
| Target LP | 1 | `time_limit_unresolved` | 56 |
| Target LP | 2 | `adversarial_example_found_or_best_known` | 2 |
| Target LP | 2 | `certified_no_adversarial_example` | 42 |
| Target LP | 2 | `time_limit_unresolved` | 56 |
| Backward score-gap screen | 1 | `adversarial_example_found_or_best_known` | 16 |
| Backward score-gap screen | 1 | `certified_no_adversarial_example` | 43 |
| Backward score-gap screen | 1 | `time_limit_unresolved` | 41 |
| Backward score-gap screen | 2 | `adversarial_example_found_or_best_known` | 16 |
| Backward score-gap screen | 2 | `certified_no_adversarial_example` | 43 |
| Backward score-gap screen | 2 | `time_limit_unresolved` | 41 |

## Outcome changes

Cross-method rows compare Target LP to Backward score-gap screen. Repeat rows compare observation 1 to observation 2 within the named method.

| Comparison | Transition | Count | Image IDs |
| :--- | :--- | ---: | :--- |
| outcome_observation_1 | `time_limit_unresolved -> adversarial_example_found_or_best_known` | 14 | 2, 9, 19, 34, 39, 44, 60, 62, 63, 64, 65, 88, 93, 98 |
| outcome_observation_1 | `time_limit_unresolved -> certified_no_adversarial_example` | 1 | 77 |
| outcome_observation_2 | `time_limit_unresolved -> adversarial_example_found_or_best_known` | 14 | 2, 9, 19, 34, 39, 44, 60, 62, 63, 64, 65, 88, 93, 98 |
| outcome_observation_2 | `time_limit_unresolved -> certified_no_adversarial_example` | 1 | 77 |
| outcome_repeat_target_lp | None | 0 | — |
| outcome_repeat_screen | None | 0 | — |

Opposing definitive outcomes across either observation: none.

Bootstrap intervals describe image-mix sensitivity. Repeat differences measure run-to-run variation separately.
