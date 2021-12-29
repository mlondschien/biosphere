use crate::tree::DecisionTree;
use crate::utils::{
    argsort, oob_samples_from_weights, sample_indices_from_weights, sample_weights,
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

pub struct RandomForest<'a> {
    pub X: &'a ArrayView2<'a, f64>,
    pub y: &'a ArrayView1<'a, f64>,
    pub n_trees: u16,
    pub max_depth: Option<u16>,
    pub mtry: u16,
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
        mtry: Option<u16>,
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
            mtry: mtry.unwrap_or((X.ncols() as f64).sqrt().floor() as u16),
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
        let y_sum = self.y.sum();

        let indices: Vec<Vec<usize>> = (0..self.X.ncols())
            .map(|col| argsort(&self.X.column(col)))
            .collect();

        for _ in 0..self.n_trees {
            let seed: u64 = rng.gen();
            let weights = sample_weights(n, &mut rng);
            let result = predict_with_tree(
                self.X,
                self.y,
                weights,
                &indices,
                y_sum,
                self.mtry,
                self.max_depth,
                self.min_samples_split,
                self.min_gain_to_split,
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

    // fn sample_features(&self, rng: &mut impl Rng) -> Vec<usize> {
    //     (0..self.X.ncols())
    //         .into_iter()
    //         .choose_multiple(rng, self.mtry)
    // }
}

#[allow(clippy::too_many_arguments)]
fn predict_with_tree<'b>(
    X: &'b ArrayView2<'b, f64>,
    y: &'b ArrayView1<'b, f64>,
    weights: Vec<usize>,
    indices: &[Vec<usize>],
    y_sum: f64,
    mtry: u16,
    max_depth: Option<u16>,
    min_samples_split: Option<usize>,
    min_gain_to_split: Option<f64>,
    seed: u64,
) -> Vec<(Vec<usize>, f64)> {
    let samples = sample_indices_from_weights(&weights, indices);
    let mut oob_samples = oob_samples_from_weights(&weights);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tree = DecisionTree::new(
        X,
        y,
        samples,
        max_depth,
        mtry,
        min_samples_split,
        min_gain_to_split,
    );

    tree.split(
        0,
        X.nrows(),
        &mut oob_samples,
        vec![false; X.ncols()],
        0,
        y_sum,
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

        let forest = RandomForest::new(&X, &y, None, Some(8), None, None, None, None);

        let predictions = forest.predict();
        let mse = (predictions - y).mapv(|x| x * x).sum();
        assert!(mse < 0.1, "mse {}", mse);
    }
}
