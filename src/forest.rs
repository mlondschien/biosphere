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
    n_trees: usize,
    seed: u64,
}

impl RandomForestParameters {
    pub fn new(
        n_trees: usize,
        seed: u64,
        max_depth: Option<usize>,
        mtry: Option<usize>,
        min_samples_leaf: usize,
        min_samples_split: usize,
    ) -> Self {
        RandomForestParameters {
            decision_tree_parameters: DecisionTreeParameters::new(
                max_depth,
                mtry,
                min_samples_split,
                min_samples_leaf,
                0,
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

    pub fn with_n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = n_trees;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.decision_tree_parameters = self.decision_tree_parameters.with_max_depth(max_depth);
        self
    }

    pub fn with_mtry(mut self, mtry: Option<usize>) -> Self {
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

pub struct RandomForest {
    random_forest_parameters: RandomForestParameters,
    trees: Vec<DecisionTree>,
}

impl RandomForest {
    pub fn new(random_forest_parameters: RandomForestParameters) -> Self {
        RandomForest {
            random_forest_parameters,
            trees: Vec::new(),
        }
    }

    pub fn default() -> Self {
        RandomForest::new(RandomForestParameters::default())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Array1<f64> {
        let mut predictions = Array1::<f64>::zeros(X.nrows());

        for tree in &self.trees {
            predictions = predictions + tree.predict(X);
        }

        predictions / self.trees.len() as f64
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) {
        let indices: Vec<Vec<usize>> = (0..X.ncols()).map(|idx| argsort(&X.column(idx))).collect();

        let mut rng = StdRng::seed_from_u64(self.random_forest_parameters.seed);

        for _ in 0..self.random_forest_parameters.n_trees {
            let mut tree = DecisionTree::new(
                self.random_forest_parameters
                    .decision_tree_parameters
                    .clone()
                    .with_seed(rng.gen()),
            );

            let weights = sample_weights(X.nrows(), &mut rng);
            let mut samples = sample_indices_from_weights(&weights, &indices);

            let mut references_to_samples = Vec::<&mut [usize]>::with_capacity(samples.len());

            // TODO: This is a hack to make the borrow checker happy.
            for sample in samples.iter_mut() {
                references_to_samples.push(sample);
            }

            tree.fit_with_sorted_samples(X, y, references_to_samples);

            self.trees.push(tree);
        }
    }

    pub fn fit_predict_oob(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let indices: Vec<Vec<usize>> = (0..X.ncols()).map(|idx| argsort(&X.column(idx))).collect();

        let mut rng = StdRng::seed_from_u64(self.random_forest_parameters.seed);

        let mut oob_predictions = Array1::<f64>::zeros(X.nrows());
        let mut oob_n_trees = Array1::<usize>::zeros(X.nrows());

        for _ in 0..self.random_forest_parameters.n_trees {
            let mut tree = DecisionTree::new(
                self.random_forest_parameters
                    .decision_tree_parameters
                    .clone()
                    .with_seed(rng.gen()),
            );

            let weights = sample_weights(X.nrows(), &mut rng);
            let mut samples = sample_indices_from_weights(&weights, &indices);

            let samples_as_slices = samples.iter_mut().map(|x| x.as_mut_slice()).collect();

            tree.fit_with_sorted_samples(X, y, samples_as_slices);

            let oob_samples = oob_samples_from_weights(&weights);
            for oob_sample in oob_samples.into_iter() {
                oob_predictions[oob_sample] += tree.predict_row(&X.row(oob_sample));
                oob_n_trees[oob_sample] += 1;
            }

            self.trees.push(tree);
        }

        oob_predictions * oob_n_trees.mapv(|x| 1. / x as f64)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::testing::load_iris;
//     use ndarray::s;

//     #[test]
//     fn test_random_forest_predict() {
//         let data = load_iris();
//         let X = data.slice(s![0..100, 0..4]);
//         let y = data.slice(s![0..100, 4]);

//         let random_forest_parameters = RandomForestParameters::default();
//         let forest = RandomForest::new(&X, &y, random_forest_parameters);

//         let predictions = forest.predict();
//         let mse = (&predictions - &y).mapv(|x| x * x).sum();
//         assert!(
//             mse < 0.1,
//             "mse {} \ny={:?}\npredictions={:?}",
//             mse,
//             y,
//             predictions
//         );
//     }
// }
