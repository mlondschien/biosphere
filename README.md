# Biosphere

Simple, fast random forests.

Random forests with a runtime of `O(n d log(n) + n_trees d n max_depth)` instead of `O(n_tree mtry n log(n) max_depth)`.

`biosphere` is available as a rust crate and as a Python package.

## Benchmarks

Ran on an M1 Pro with `n_jobs=4`. Wall-time to fit a Random Forest including OOB score with 400 trees to
the [NYC Taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
minimum over 10 runs. After feature engineering, the dataset consists of 5 numerical and 7 one-hot encoded
features.

| model        | 1000   | 2000   | 4000   | 8000   | 16000   | 32000   | 64000   | 128000   | 256000   | 512000   | 1024000   | 2048000   |
|:-------------|:-------|:-------|:-------|:-------|:--------|:--------|:--------|:---------|:---------|:---------|:----------|:----------|
| biosphere    | 0.04s  | 0.08s  | 0.15s  | 0.32s  | 0.65s   | 1.40s   | 2.97s   | 6.48s    | 15.53s   | 37.91s   | 96.69s    | 231.82s   |
| scikit-learn | 0.28s  | 0.34s  | 0.46s  | 0.69s  | 1.23s   | 2.47s   | 4.99s   | 10.49s   | 22.11s   | 51.04s   | 118.95s   | 271.03s   |