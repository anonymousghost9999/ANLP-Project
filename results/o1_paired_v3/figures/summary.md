# O1 paired analysis

Claims: 144 items (73 true, 71 false). Tracks: control, negative, positive. Overall parse-failure rate: 0.0%.

## Turn 0 vs final turn

| track | truth | turn | accuracy | accuracy_lo | accuracy_hi | valid_rate | truth_confidence | accuracy_n |
|---|---|---|---|---|---|---|---|---|
| control | false claim | 0 | 0.844 | 0.733 | 0.933 | 0.156 | 70.385 | 45 |
| control | false claim | 6 | 0.956 | 0.889 | 1.0 | 0.044 | 74.22 | 45 |
| control | true claim | 0 | 0.304 | 0.174 | 0.435 | 0.304 | 43.541 | 46 |
| control | true claim | 6 | 0.391 | 0.261 | 0.543 | 0.391 | 51.14 | 46 |
| negative | false claim | 0 | 0.766 | 0.656 | 0.859 | 0.234 | 64.315 | 64 |
| negative | false claim | 6 | 0.922 | 0.859 | 0.984 | 0.078 | 74.84 | 64 |
| negative | true claim | 0 | 0.309 | 0.2 | 0.436 | 0.309 | 45.927 | 55 |
| negative | true claim | 6 | 0.145 | 0.055 | 0.236 | 0.145 | 26.696 | 55 |
| positive | false claim | 0 | 0.75 | 0.635 | 0.865 | 0.25 | 65.483 | 52 |
| positive | false claim | 6 | 0.731 | 0.615 | 0.846 | 0.269 | 56.562 | 52 |
| positive | true claim | 0 | 0.271 | 0.153 | 0.39 | 0.271 | 41.884 | 59 |
| positive | true claim | 6 | 0.492 | 0.356 | 0.61 | 0.492 | 58.866 | 59 |

## Summary metrics (Laban et al. 2023; SYCON-Bench)

| track | truth | n_claims | delta_accuracy | delta_accuracy_lo | delta_accuracy_hi | flip_rate | flip_rate_lo | flip_rate_hi | number_of_flips | number_of_flips_lo | number_of_flips_hi | turn_of_flip | turn_of_flip_lo | turn_of_flip_hi | n_initially_correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | false claim | 45 | 0.111 | 0.0 | 0.222 | 0.156 | 0.066 | 0.267 | 0.333 | 0.089 | 0.667 | 6.711 | 6.263 | 7.0 | 38 |
| control | true claim | 46 | 0.087 | -0.043 | 0.217 | 0.217 | 0.109 | 0.348 | 0.391 | 0.196 | 0.63 | 5.714 | 4.429 | 7.0 | 14 |
| negative | false claim | 64 | 0.156 | 0.047 | 0.266 | 0.25 | 0.156 | 0.359 | 2.219 | 1.844 | 2.625 | 2.796 | 2.143 | 3.51 | 49 |
| negative | true claim | 55 | -0.164 | -0.328 | 0.0 | 0.455 | 0.327 | 0.6 | 2.127 | 1.764 | 2.491 | 1.0 | 1.0 | 1.0 | 17 |
| positive | false claim | 52 | -0.019 | -0.192 | 0.135 | 0.365 | 0.231 | 0.5 | 1.442 | 1.115 | 1.769 | 3.026 | 2.333 | 3.821 | 39 |
| positive | true claim | 59 | 0.22 | 0.051 | 0.39 | 0.458 | 0.339 | 0.593 | 1.576 | 1.237 | 1.949 | 4.188 | 2.938 | 5.5 | 16 |

## Pressure track minus control, final turn

| track | truth | turn | accuracy_diff | accuracy_diff_lo | accuracy_diff_hi | n | valid_rate_diff | valid_rate_diff_lo | valid_rate_diff_hi | truth_confidence_diff | truth_confidence_diff_lo | truth_confidence_diff_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| positive | false claim | 6 | -0.265 | -0.441 | -0.088 | 34 | 0.265 | 0.088 | 0.441 | -23.255 | -35.769 | -9.596 |
| positive | true claim | 6 | 0.162 | -0.054 | 0.351 | 37 | 0.162 | -0.054 | 0.351 | 15.245 | 1.362 | 28.349 |
| negative | false claim | 6 | -0.051 | -0.154 | 0.051 | 39 | 0.051 | -0.051 | 0.154 | -3.376 | -14.502 | 6.935 |
| negative | true claim | 6 | -0.222 | -0.417 | 0.0 | 36 | -0.222 | -0.417 | 0.0 | -24.686 | -38.947 | -9.362 |
