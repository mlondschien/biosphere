use crate::tree::DecisionTree;
use crate::utils::{
    argsort, oob_samples_from_weights, sample_indices_from_weights, sample_weights,
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::Rng;
use rand::SeedableRng;

pub struct RandomForest<'a> {
    pub X: &'a ArrayView2<'a, f64>,
    pub y: &'a ArrayView1<'a, f64>,
    pub n_trees: u16,
    pub max_depth: Option<u16>,
    pub mtry: usize,
    pub min_samples_split: Option<usize>,
    pub min_gain_to_split: Option<f64>,
    pub seed: u64,
}

impl<'a> RandomForest<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        X: &'a ArrayView2<'a, f64>,
        y: &'a ArrayView1<'a, f64>,
        n_trees: Option<u16>,
        max_depth: Option<u16>,
        mtry: Option<usize>,
        min_samples_split: Option<usize>,
        min_gain_to_split: Option<f64>,
        seed: Option<u64>,
    ) -> Self {
        RandomForest {
            X,
            y,
            n_trees: n_trees.unwrap_or(100),
            max_depth,
            min_samples_split,
            min_gain_to_split,
            mtry: mtry.unwrap_or((X.ncols() as f64).sqrt().ceil() as usize),
            seed: seed.unwrap_or(0),
        }
    }

    pub fn default(X: &'a ArrayView2<'a, f64>, y: &'a ArrayView1<'a, f64>) -> Self {
        RandomForest::new(X, y, None, None, None, None, None, None)
    }

    pub fn predict(&self) -> Array1<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);

        let n = self.X.nrows();
        let mut predictions = Array1::zeros(self.y.len());
        let mut n_predictions = Array1::<u32>::zeros(self.y.len());

        let mut indices: Vec<Vec<usize>> = Vec::with_capacity(self.X.ncols());
        for i in 0..self.X.ncols() {
            indices.push(argsort(&self.X.column(i).to_vec()));
        }

        for _ in 0..self.n_trees {
            let weights = sample_weights(n, &mut rng);
            let features = self.sample_features(&mut rng);
            let result = predict_with_tree(
                self.X,
                self.y,
                &weights,
                &indices,
                &features,
                self.max_depth,
                self.min_samples_split,
                self.min_gain_to_split,
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

    fn sample_features(&self, rng: &mut impl Rng) -> Vec<usize> {
        (0..self.X.ncols())
            .into_iter()
            .choose_multiple(rng, self.mtry)
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_with_tree<'b>(
    X: &'b ArrayView2<'b, f64>,
    y: &'b ArrayView1<'b, f64>,
    weights: &[usize],
    indices: &[Vec<usize>],
    features: &[usize],
    max_depth: Option<u16>,
    min_samples_split: Option<usize>,
    min_gain_to_split: Option<f64>,
) -> Vec<(Vec<usize>, f64)> {
    let samples = sample_indices_from_weights(weights, indices, features);
    let mut oob_samples = oob_samples_from_weights(weights);

    let tree = DecisionTree::new(X, y, max_depth, min_samples_split, min_gain_to_split);

    tree.split(samples, &mut oob_samples, features, 0)
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

        let forest = RandomForest::default(&X, &y);

        let predictions = forest.predict();
        assert!((predictions - y).mapv(|x| x * x).sum() < 0.1);
    }
}
