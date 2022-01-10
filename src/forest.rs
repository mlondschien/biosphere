use crate::tree::{DecisionTree, DecisionTreeParameters};
use crate::utils::{
    argsort, oob_samples_from_weights, sample_indices_from_weights, sample_weights,
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::ThreadPoolBuilder;

#[derive(Clone)]
pub struct RandomForestParameters {
    decision_tree_parameters: DecisionTreeParameters,
    n_trees: usize,
    seed: u64,
    n_jobs: Option<usize>,
}

impl RandomForestParameters {
    pub fn new(
        n_trees: usize,
        seed: u64,
        max_depth: Option<usize>,
        mtry: Option<usize>,
        min_samples_leaf: usize,
        min_samples_split: usize,
        n_jobs: Option<usize>,
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
            n_jobs,
        }
    }

    pub fn default() -> Self {
        RandomForestParameters {
            decision_tree_parameters: DecisionTreeParameters::default(),
            n_trees: 100,
            seed: 0,
            n_jobs: None,
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

    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
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
        let mut thread_pool_builder = ThreadPoolBuilder::new();

        if let Some(n_jobs) = self.random_forest_parameters.n_jobs {
            thread_pool_builder = thread_pool_builder.num_threads(n_jobs);
        }

        let thread_pool = thread_pool_builder.build().unwrap();

        let indices: Vec<usize> = (0..X.ncols()).collect();
        let indices: Vec<Vec<usize>> = thread_pool.install(|| {
            indices
                .into_par_iter()
                .map(|idx| argsort(&X.column(idx)))
                .collect()
        });

        let mut rng = StdRng::seed_from_u64(self.random_forest_parameters.seed);
        let seeds: Vec<u64> = (0..self.random_forest_parameters.n_trees)
            .into_iter()
            .map(|_| rng.gen())
            .collect();

        self.trees = thread_pool.install(|| {
            seeds
                .into_par_iter()
                .map(|seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut tree = DecisionTree::new(
                        self.random_forest_parameters
                            .decision_tree_parameters
                            .clone()
                            .with_seed(seed),
                    );

                    let weights = sample_weights(X.nrows(), &mut rng);
                    let mut samples = sample_indices_from_weights(&weights, &indices);

                    let mut references_to_samples =
                        Vec::<&mut [usize]>::with_capacity(samples.len());

                    // fit_with_sorted_samples expects Vec<&[usize]>. This could be done more
                    // elegantly.
                    for sample in samples.iter_mut() {
                        references_to_samples.push(sample);
                    }

                    tree.fit_with_sorted_samples(X, y, references_to_samples);
                    tree
                })
                .collect()
        })
    }

    pub fn fit_predict_oob(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let mut thread_pool_builder = ThreadPoolBuilder::new();

        if let Some(n_jobs) = self.random_forest_parameters.n_jobs {
            thread_pool_builder = thread_pool_builder.num_threads(n_jobs);
        }

        let thread_pool = thread_pool_builder.build().unwrap();

        let indices: Vec<usize> = (0..X.ncols()).collect();

        let mut rng = StdRng::seed_from_u64(self.random_forest_parameters.seed);
        let seeds: Vec<u64> = (0..self.random_forest_parameters.n_trees)
            .into_iter()
            .map(|_| rng.gen())
            .collect();

        let tree_parameters = self
            .random_forest_parameters
            .decision_tree_parameters
            .clone();

        let result: Vec<(DecisionTree, Vec<usize>, Vec<f64>)> = thread_pool.scope(move |_| {
            let indices: Vec<Vec<usize>> = indices
                .into_par_iter()
                .map(|idx| argsort(&X.column(idx)))
                .collect();

            seeds
                .into_par_iter()
                .map(move |seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut tree = DecisionTree::new(tree_parameters.clone().with_seed(rng.gen()));

                    let weights = sample_weights(X.nrows(), &mut rng);
                    let mut samples = sample_indices_from_weights(&weights, &indices);
                    let oob_samples = oob_samples_from_weights(&weights);

                    let samples_as_slices = samples.iter_mut().map(|x| x.as_mut_slice()).collect();

                    tree.fit_with_sorted_samples(X, y, samples_as_slices);

                    let mut oob_predictions = Vec::<f64>::with_capacity(oob_samples.len());
                    for sample in oob_samples.iter() {
                        oob_predictions.push(tree.predict_row(&X.row(*sample)));
                    }

                    (tree, oob_samples, oob_predictions)
                })
                .collect()
        });

        let mut oob_predictions: Array1<f64> = Array1::zeros(X.nrows());
        let mut oob_n_trees: Array1<usize> = Array1::zeros(X.nrows());

        for (tree, oob_samples, oob_predictions_) in result {
            self.trees.push(tree);
            for (idx, prediction) in oob_samples.into_iter().zip(oob_predictions_.into_iter()) {
                oob_predictions[idx] += prediction;
                oob_n_trees[idx] += 1;
            }
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
