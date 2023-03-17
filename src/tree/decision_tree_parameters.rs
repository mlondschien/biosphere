#[derive(Clone, Debug, Default)]
pub enum MaxFeatures {
    // Consider all `d` features at each split.
    #[default]
    None,
    // Compute `int(fraction * d)` features at each split.
    Fraction(f64),
    // Consider `value` features at each split.
    Value(usize),
    // Consider `int(sqrt(d))` features at each split.
    Sqrt,
    // Consider `callable(d)` features at each split.
    Callable(fn(usize) -> usize),
}

impl MaxFeatures {
    pub fn from_n_features(&self, n_features: usize) -> usize {
        let value = match self {
            MaxFeatures::None => n_features,
            MaxFeatures::Fraction(fraction) => (fraction * n_features as f64) as usize,
            MaxFeatures::Value(number) => *number,
            MaxFeatures::Sqrt => (n_features as f64).sqrt() as usize,
            MaxFeatures::Callable(callable) => callable(n_features),
        };

        value.max(1).min(n_features)
    }
}

#[derive(Clone, Debug)]
pub struct DecisionTreeParameters {
    // Maximum depth of the tree. If `None`, nodes are expanded until all leaves are
    // pure or contain fewer than `min_samples_split` samples.
    pub max_depth: Option<usize>,
    // The number of features to consider when looking for the best split.
    pub max_features: MaxFeatures,
    // Minimum number of samples required to split a node.
    pub min_samples_split: usize,
    // The minimum number of samples required to be at a leaf node.
    pub min_samples_leaf: usize,
    // Seed for reproducibility.
    pub random_state: u64,
}

impl Default for DecisionTreeParameters {
    fn default() -> Self {
        DecisionTreeParameters {
            max_depth: None,
            max_features: MaxFeatures::default(),
            min_samples_split: 2,
            min_samples_leaf: 1,
            random_state: 0,
        }
    }
}

impl DecisionTreeParameters {
    pub fn new(
        max_depth: Option<usize>,
        max_features: MaxFeatures,
        min_samples_split: usize,
        min_samples_leaf: usize,
        random_state: u64,
    ) -> Self {
        DecisionTreeParameters {
            max_depth,
            max_features,
            min_samples_split,
            min_samples_leaf,
            random_state,
        }
    }

    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    #[case(MaxFeatures::None, 10, 10)]
    #[case(MaxFeatures::Fraction(1.), 10, 10)]
    #[case(MaxFeatures::Fraction(0.8), 10, 8)]
    #[case(MaxFeatures::Fraction(0.01), 10, 1)]
    #[case(MaxFeatures::Fraction(2.), 10, 10)]
    #[case(MaxFeatures::Value(5), 10, 5)]
    #[case(MaxFeatures::Sqrt, 10, 3)]
    #[case(MaxFeatures::Callable(|x| x % 4), 10, 2)]
    fn test_MaxFeatures(
        #[case] max_features: MaxFeatures,
        #[case] n_features: usize,
        #[case] expected: usize,
    ) {
        assert_eq!(max_features.from_n_features(n_features), expected);
    }
}
