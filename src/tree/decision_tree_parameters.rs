#[derive(Clone)]
pub struct DecisionTreeParameters {
    // Maximum depth of the tree.
    pub max_depth: Option<usize>,
    pub mtry: Option<usize>,
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
            mtry: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            seed: 0,
        }
    }

    pub fn new(
        max_depth: Option<usize>,
        mtry: Option<usize>,
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

    pub fn with_mtry(mut self, mtry: Option<usize>) -> Self {
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
