# O1 paired analysis

Claims: 92 items (70 true, 22 false). Tracks: negative, positive. Overall parse-failure rate: 1.3%.

## Turn 0 vs final turn

| track | truth | turn | accuracy | accuracy_lo | accuracy_hi | valid_rate | truth_confidence | accuracy_n |
|---|---|---|---|---|---|---|---|---|
| negative | false claim | 0 | 0.636 | 0.453 | 0.818 | 0.364 | 56.754 | 22 |
| negative | false claim | 2 | 0.682 | 0.5 | 0.864 | 0.318 | 69.321 | 22 |
| negative | true claim | 0 | 0.594 | 0.478 | 0.696 | 0.594 | 59.489 | 69 |
| negative | true claim | 2 | 0.203 | 0.116 | 0.304 | 0.203 | 27.732 | 69 |
| positive | false claim | 0 | 0.636 | 0.453 | 0.818 | 0.364 | 56.754 | 22 |
| positive | false claim | 2 | 0.409 | 0.182 | 0.636 | 0.591 | 43.936 | 22 |
| positive | true claim | 0 | 0.594 | 0.478 | 0.696 | 0.594 | 59.489 | 69 |
| positive | true claim | 2 | 0.721 | 0.618 | 0.824 | 0.721 | 75.2 | 68 |

## Summary metrics (Laban et al. 2023; SYCON-Bench)

| track | truth | n_claims | delta_accuracy | delta_accuracy_lo | delta_accuracy_hi | flip_rate | flip_rate_lo | flip_rate_hi | number_of_flips | number_of_flips_lo | number_of_flips_hi | turn_of_flip | turn_of_flip_lo | turn_of_flip_hi | n_initially_correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| negative | false claim | 22 | 0.045 | -0.273 | 0.364 | 0.591 | 0.364 | 0.773 | 0.864 | 0.591 | 1.136 | 2.071 | 1.643 | 2.5 | 14 |
| negative | true claim | 70 | -0.391 | -0.536 | -0.232 | 0.586 | 0.471 | 0.7 | 0.957 | 0.814 | 1.1 | 1.171 | 1.049 | 1.293 | 41 |
| positive | false claim | 22 | -0.227 | -0.5 | 0.0 | 0.409 | 0.227 | 0.636 | 0.591 | 0.318 | 0.864 | 2.0 | 1.571 | 2.429 | 14 |
| positive | true claim | 70 | 0.118 | 0.0 | 0.235 | 0.257 | 0.157 | 0.357 | 0.343 | 0.214 | 0.486 | 2.78 | 2.585 | 2.927 | 41 |
