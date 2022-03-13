# Changelog

## 0.3.0 - (2022-03-13)

- **New features:**
    - The `mtry` parameter for classes `DecisionTree` and `RandomForest` can now be supplied with a fraction, an integer, `None` and `"sqrt"`.

## 0.2.2 - (2022-02-22)

- **Other changes:**
    - Speedup of `DecisisionTreeNode.split_samples` resulting in overall 6 - 20% faster
      tree fitting.

## 0.2.1 - (2022-01-13)

- **Bug fixes:**
    - `DecisionTreeNode` no longer returns wrong leaf value if splitting is stopped
      due to `min_samples_split`.  

## 0.2.0 - (2022-01-11)

**New features:**
    - Parallization for `RandomForest::fit` and `RandomForest::fit_predict_oob`.