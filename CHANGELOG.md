# Changelog

## 0.2.1 - (2022-01-13)

- **Bug fixes:**
    - `DecisionTreeNode` no longer returns wrong leaf value if splitting is stopped
      due to `min_samples_split`.  

## 0.2.0 - (2022-01-11)

**New features:**
    - Parallization for `RandomForest::fit` and `RandomForest::fit_predict_oob`.