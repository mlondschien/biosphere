use crate::tree::DecisionTree;
use crate::utils::{
    argsort, oob_samples_from_weights, sample_indices_from_weights, sample_weights,
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::SmallRng;
use rand::seq::IteratorRandom;
use rand::Rng;
use rand::SeedableRng;

struct RandomForest<'a> {
    X: &'a ArrayView2<'a, f64>,
    y: &'a ArrayView1<'a, f64>,
    n_trees: usize,
    max_depth: usize,
    mtry: usize,
    seed: u64,
}

impl<'a> RandomForest<'a> {
    #[allow(dead_code)]
    fn predict(&self) -> Array1<f64> {
        let mut rng = SmallRng::seed_from_u64(self.seed);

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

fn predict_with_tree<'b>(
    X: &'b ArrayView2<'b, f64>,
    y: &'b ArrayView1<'b, f64>,
    weights: &[usize],
    indices: &[Vec<usize>],
    features: &[usize],
    max_depth: usize,
) -> Vec<(Vec<usize>, f64)> {
    let samples = sample_indices_from_weights(weights, indices, features);
    let mut oob_samples = oob_samples_from_weights(weights);

    let tree = DecisionTree { X, y, max_depth };

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

        let forest = RandomForest {
            X: &X,
            y: &y,
            n_trees: 100,
            max_depth: 3,
            mtry: 3,
            seed: 7,
        };

        let predictions = forest.predict();
        assert!((predictions - y).mapv(|x| x * x).sum() < 0.1);
    }
}
