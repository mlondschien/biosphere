# Changelog

## 0.4.0 - (2025-04-26)

- **Other changes:**
    - Updated `ndarray` to 0.16. Thanks @Havunen!

## 0.3.0 - (2022-03-14)

- **Breaking changes:**
    - The arguments `mtry` and `seed` to `DecisionTree` and `RandomForest` have been renamed to `max_features` and `random_state`, aligning them with their scikit-learn counterparts.
    - Supplying `n_jobs=None` will now result in no parallelization, aligning its behaviour with scikit-learn. To use all processes, use `n_jobs=-1`.

- **New features:**
    - The `max_features` parameter for classes `DecisionTree` and `RandomForest` can now be supplied with a fraction, an integer, `None` and `"sqrt"`.

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