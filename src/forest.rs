use crate::tree::{DecisionTree, DecisionTreeParameters, MaxFeatures};
use crate::utils::{
    argsort, oob_samples_from_weights, sample_indices_from_weights, sample_weights,
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Configuration for a [`RandomForest`].
///
/// Use the builder methods to customise the forest. Most users only need to
/// change `n_estimators` and `seed`; all other parameters have sensible defaults.
///
/// ```rust
/// use biosphere::{RandomForestParameters, MaxFeatures};
///
/// let params = RandomForestParameters::default()
///     .with_n_estimators(200)
///     .with_seed(42)
///     .with_max_depth(Some(10))
///     .with_max_features(MaxFeatures::Sqrt)
///     .with_n_jobs(Some(-1)); // use all CPU cores for training
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct RandomForestParameters {
    decision_tree_parameters: DecisionTreeParameters,
    n_estimators: usize,
    seed: u64,
    // The number of jobs to run in parallel for `fit` and `fit_predict_oob`.
    // `None` means 1. `-1` means using all processors.
    n_jobs: Option<i32>,
}

impl Default for RandomForestParameters {
    fn default() -> Self {
        RandomForestParameters {
            decision_tree_parameters: DecisionTreeParameters::default(),
            n_estimators: 100,
            seed: 0,
            n_jobs: None,
        }
    }
}

impl RandomForestParameters {
    pub fn new(
        n_estimators: usize,
        seed: u64,
        max_depth: Option<usize>,
        max_features: MaxFeatures,
        min_samples_leaf: usize,
        min_samples_split: usize,
        n_jobs: Option<i32>,
    ) -> Self {
        RandomForestParameters {
            decision_tree_parameters: DecisionTreeParameters::new(
                max_depth,
                max_features,
                min_samples_split,
                min_samples_leaf,
                0,
            ),
            n_estimators,
            seed,
            n_jobs,
        }
    }

    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
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

    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.decision_tree_parameters = self
            .decision_tree_parameters
            .with_max_features(max_features);
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

    pub fn with_n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

/// A random forest ensemble for regression or binary classification.
///
/// Trains many [`DecisionTree`]s on bootstrap samples of your data and averages
/// their predictions. More trees reduce variance at diminishing returns; 100–500
/// is usually enough.
///
/// ```rust
/// use biosphere::{RandomForest, RandomForestParameters};
/// use ndarray::array;
///
/// let X = array![[0.0, 1.0], [1.0, 0.0]];
/// let y = array![0.0, 1.0];
///
/// let mut forest = RandomForest::new(RandomForestParameters::default());
/// forest.fit(&X.view(), &y.view());
///
/// let predictions = forest.predict(&X.view()); // Array1<f64>, one value per row
/// ```
///
/// For GPU inference, convert to a [`FlatForest`] first.
///
/// [`FlatForest`]: crate::FlatForest
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct RandomForest {
    random_forest_parameters: RandomForestParameters,
    trees: Vec<DecisionTree>,
}

fn build_thread_pool(n_jobs: Option<i32>) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    let n_threads = match n_jobs {
        Some(n) if n >= 1 => Some(n as usize),
        Some(_) => None, // -1 = all cores
        None => Some(1),
    };
    if let Some(n) = n_threads {
        builder = builder.num_threads(n);
    }
    builder.build().expect("failed to build rayon thread pool")
}

impl Default for RandomForest {
    fn default() -> Self {
        RandomForest::new(RandomForestParameters::default())
    }
}

impl RandomForest {
    pub fn new(random_forest_parameters: RandomForestParameters) -> Self {
        RandomForest {
            random_forest_parameters,
            trees: Vec::new(),
        }
    }

    pub(crate) fn trees(&self) -> &[DecisionTree] {
        &self.trees
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Array1<f64> {
        let mut predictions = Array1::<f64>::zeros(X.nrows());
        let mut scratch = Array1::<f64>::zeros(X.nrows());

        for tree in &self.trees {
            tree.predict_into(X, &mut scratch);
            predictions += &scratch;
        }

        predictions / self.trees.len() as f64
    }

    /// Fit the forest on training data.
    ///
    /// **Reproducibility note (v0.5):** Each tree's feature-selection seed is now derived
    /// directly from its bootstrap seed rather than from a secondary RNG draw. Forests
    /// trained with the same seed as v0.4.x will produce different trees starting from
    /// v0.5.0.
    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) {
        let thread_pool = build_thread_pool(self.random_forest_parameters.n_jobs);

        let indices: Vec<usize> = (0..X.ncols()).collect();
        let indices: Vec<Vec<usize>> = thread_pool.install(|| {
            indices
                .into_par_iter()
                .map(|idx| argsort(&X.column(idx)))
                .collect()
        });

        let mut rng = StdRng::seed_from_u64(self.random_forest_parameters.seed);
        let seeds: Vec<u64> = (0..self.random_forest_parameters.n_estimators)
            .map(|_| rng.random::<u64>())
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
                            .with_random_state(seed),
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

    /// Fit the forest and return out-of-bag predictions for each training sample.
    ///
    /// Each element of the returned array is the average prediction of the trees
    /// for which that sample was out-of-bag. Samples that were in-bag for every
    /// estimator (i.e. never left out during bootstrap sampling) will have
    /// `f64::NAN` as their OOB prediction.
    ///
    /// **Reproducibility note (v0.5):** Each tree's feature-selection seed is now derived
    /// directly from its bootstrap seed rather than from a secondary RNG draw. Forests
    /// trained with the same seed as v0.4.x will produce different trees starting from
    /// v0.5.0.
    pub fn fit_predict_oob(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let thread_pool = build_thread_pool(self.random_forest_parameters.n_jobs);

        let indices: Vec<usize> = (0..X.ncols()).collect();

        let mut rng = StdRng::seed_from_u64(self.random_forest_parameters.seed);
        let seeds: Vec<u64> = (0..self.random_forest_parameters.n_estimators)
            .map(|_| rng.random::<u64>())
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
                    let mut tree =
                        DecisionTree::new(tree_parameters.clone().with_random_state(seed));

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
        let mut oob_n_estimators: Array1<usize> = Array1::zeros(X.nrows());

        for (tree, oob_samples, oob_predictions_) in result {
            self.trees.push(tree);
            for (idx, prediction) in oob_samples.into_iter().zip(oob_predictions_.into_iter()) {
                oob_predictions[idx] += prediction;
                oob_n_estimators[idx] += 1;
            }
        }

        Array1::from_iter(
            oob_predictions
                .iter()
                .zip(oob_n_estimators.iter())
                .map(
                    |(&pred, &n)| {
                        if n == 0 { f64::NAN } else { pred / n as f64 }
                    },
                ),
        )
    }
}
