#[derive(Clone)]
pub enum Mtry {
    None,
    Fraction(f64),
    Value(usize),
    Sqrt,
    Callable(fn(usize) -> usize),
}

impl Mtry {
    pub fn mtry(&self, n_features: usize) -> usize {
        let value = match self {
            Mtry::None => n_features,
            Mtry::Fraction(fraction) => (fraction * n_features as f64) as usize,
            Mtry::Value(number) => *number,
            Mtry::Sqrt => (n_features as f64).sqrt() as usize,
            Mtry::Callable(callable) => callable(n_features),
        };

        value.max(1).min(n_features)
    }
}

#[derive(Clone)]
pub struct DecisionTreeParameters {
    // Maximum depth of the tree.
    pub max_depth: Option<usize>,
    pub mtry: Mtry,
    // Minimum number of samples required to split a node.
    pub min_samples_split: usize,
    //
    pub min_samples_leaf: usize,
    //
    pub seed: u64,
}

impl DecisionTreeParameters {
    pub fn default() -> Self {
        DecisionTreeParameters {
            max_depth: None,
            mtry: Mtry::None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            seed: 0,
        }
    }

    pub fn new(
        max_depth: Option<usize>,
        mtry: Mtry,
        min_samples_split: usize,
        min_samples_leaf: usize,
        seed: u64,
    ) -> Self {
        DecisionTreeParameters {
            max_depth,
            mtry,
            min_samples_split,
            min_samples_leaf,
            seed,
        }
    }

    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn with_mtry(mut self, mtry: Mtry) -> Self {
        self.mtry = mtry;
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

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    #[case(Mtry::None, 10, 10)]
    #[case(Mtry::Fraction(1.), 10, 10)]
    #[case(Mtry::Fraction(0.8), 10, 8)]
    #[case(Mtry::Fraction(0.01), 10, 1)]
    #[case(Mtry::Fraction(2.), 10, 10)]
    #[case(Mtry::Value(5), 10, 5)]
    #[case(Mtry::Sqrt, 10, 3)]
    #[case(Mtry::Callable(|x| x % 4), 10, 2)]
    fn test_mtry(#[case] mtry: Mtry, #[case] n_features: usize, #[case] expected: usize) {
        assert_eq!(mtry.mtry(n_features), expected);
    }
}