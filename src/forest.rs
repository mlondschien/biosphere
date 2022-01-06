use crate::tree::{DecisionTree, DecisionTreeParameters};
use crate::utils::{
    argsort, oob_samples_from_weights, sample_indices_from_weights, sample_weights,
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

pub struct RandomForestParameters {
    decision_tree_parameters: DecisionTreeParameters,
    n_trees: u16,
    seed: u64,
}

impl RandomForestParameters {
    pub fn new(
        n_trees: u16,
        seed: u64,
        max_depth: Option<u16>,
        mtry: Option<u16>,
        min_samples_leaf: usize,
        min_samples_split: usize,
    ) -> Self {
        RandomForestParameters {
            decision_tree_parameters: DecisionTreeParameters::new(
                max_depth,
                mtry,
                min_samples_split,
                min_samples_leaf,
            ),
            n_trees,
            seed,
        }
    }

    pub fn default() -> Self {
        RandomForestParameters {
            decision_tree_parameters: DecisionTreeParameters::default(),
            n_trees: 100,
            seed: 0,
        }
    }

    pub fn with_n_trees(mut self, n_trees: u16) -> Self {
        self.n_trees = n_trees;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_max_depth(mut self, max_depth: Option<u16>) -> Self {
        self.decision_tree_parameters = self.decision_tree_parameters.with_max_depth(max_depth);
        self
    }

    pub fn with_mtry(mut self, mtry: Option<u16>) -> Self {
        self.decision_tree_parameters = self.decision_tree_parameters.with_mtry(mtry);
        self
    }

    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.decision_tree_parameters = self
            .decision_tree_parameters
            .with_min_samples_leaf(min_samples_leaf);
        self
    }

    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.decision_tree_parameters = self
            .decision_tree_parameters
            .with_min_samples_split(min_samples_split);
        self
    }
}

pub struct RandomForest<'a> {
    X: &'a ArrayView2<'a, f64>,
    y: &'a ArrayView1<'a, f64>,
    random_forest_parameters: RandomForestParameters,
}

impl<'a> RandomForest<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        X: &'a ArrayView2<'a, f64>,
        y: &'a ArrayView1<'a, f64>,
        random_forest_parameters: RandomForestParameters,
    ) -> Self {
        RandomForest {
            X,
            y,
            random_forest_parameters,
        }
    }

    pub fn default(X: &'a ArrayView2<'a, f64>, y: &'a ArrayView1<'a, f64>) -> Self {
        RandomForest::new(X, y, RandomForestParameters::default())
    }

    pub fn predict(&self) -> Array1<f64> {
        let mut rng = StdRng::seed_from_u64(self.random_forest_parameters.seed);

        let n = self.X.nrows();
        let mut predictions = Array1::<f64>::zeros(self.y.len());
        let mut n_predictions = Array1::<u32>::zeros(self.y.len());

        let indices: Vec<Vec<usize>> = (0..self.X.ncols())
            .map(|col| argsort(&self.X.column(col)))
            .collect();

        for _ in 0..self.random_forest_parameters.n_trees {
            let seed: u64 = rng.gen();
            let weights = sample_weights(n, &mut rng);
            let result = predict_with_tree(
                self.X,
                self.y,
                weights,
                &indices,
                self.random_forest_parameters
                    .decision_tree_parameters
                    .clone(),
                seed,
            );
            for (idxs, prediction) in result {
                for idx in idxs {
                    predictions[idx] += prediction;
                    n_predictions[idx] += 1;
                }
            }
        }

        for i in 0..n {
            predictions[i] /= n_predictions[i] as f64;
        }
        predictions
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_with_tree<'b>(
    X: &'b ArrayView2<'b, f64>,
    y: &'b ArrayView1<'b, f64>,
    weights: Vec<usize>,
    indices: &[Vec<usize>],
    decision_tree_parameters: DecisionTreeParameters,
    seed: u64,
) -> Vec<(Vec<usize>, f64)> {
    let samples = sample_indices_from_weights(&weights, indices);
    let mut oob_samples = oob_samples_from_weights(&weights);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tree = DecisionTree::new(X, y, samples, decision_tree_parameters);

    tree.split(
        0,
        X.nrows(),
        &mut oob_samples,
        vec![false; X.ncols()],
        0,
        None,
        &mut rng,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::load_iris;
    use ndarray::s;

    #[test]
    fn test_random_forest_predict() {
        let data = load_iris();
        let X = data.slice(s![0..100, 0..4]);
        let y = data.slice(s![0..100, 4]);

        let random_forest_parameters = RandomForestParameters::default();
        let forest = RandomForest::new(&X, &y, random_forest_parameters);

        let predictions = forest.predict();
        let mse = (&predictions - &y).mapv(|x| x * x).sum();
        assert!(
            mse < 0.1,
            "mse {} \ny={:?}\npredictions={:?}",
            mse,
            y,
            predictions
        );
    }
}
