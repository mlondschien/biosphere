# Biosphere

Simple, fast random forests.

Random forests with a runtime of `O(n d log(n) + n_trees d n max_depth)` instead of `O(n_tree mtry n log(n) max_depth)`.

`biosphere` is available as a rust crate and as a Python package.

## Benchmarks

Ran on a dedicated M1 Pro with `n_jobs=4`. Measured is wall-time to fit a Random Forest with 400 trees to
the [NYC Taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
After feature engineering, the dataset consists of 5 numerical and 7 one-hot encoded
features.

| model       | 1000   | 2000   | 4000   | 8000   | 16000   | 32000   | 64000   | 128000   | 256000   | 512000   | 1024000   | 2048000   |
|:------------|:-------|:-------|:-------|:-------|:--------|:--------|:--------|:---------|:---------|:---------|:----------|:----------|
| ScikitLearn | 0.25s  | 0.28s  | 0.37s  | 0.55s  | 0.98s   | 1.99s   | 4.09s   | 8.65s    | 18.77s   | 45.00s   | 107.71s   | 243.74s   |
| biosphere   | 0.04s  | 0.08s  | 0.15s  | 0.30s  | 0.62s   | 1.35s   | 2.83s   | 6.43s    | 15.43s   | 38.62s   | 101.25s   | 230.16s   |