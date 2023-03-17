use crate::tree::decision_tree_node::DecisionTreeNode;
use crate::tree::DecisionTreeParameters;
use crate::utils::sorted_samples;
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct DecisionTree {
    decision_tree_parameters: DecisionTreeParameters,
    node: DecisionTreeNode,
}

impl Default for DecisionTree {
    fn default() -> Self {
        DecisionTree::new(DecisionTreeParameters::default())
    }
}

impl DecisionTree {
    pub fn new(decision_tree_parameters: DecisionTreeParameters) -> Self {
        DecisionTree {
            decision_tree_parameters,
            node: DecisionTreeNode::default(),
        }
    }

    pub fn fit_with_samples(
        &mut self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        samples: &[usize],
    ) {
        let mut samples = sorted_samples(X, samples);
        let samples_as_slices = samples.iter_mut().map(|x| x.as_mut_slice()).collect();

        self.fit_with_sorted_samples(X, y, samples_as_slices);
    }

    pub fn fit_with_sorted_samples(
        &mut self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        samples: Vec<&mut [usize]>,
    ) {
        let mut rng = StdRng::seed_from_u64(self.decision_tree_parameters.random_state);

        let mut sum = 0.;
        for s in samples[0].iter() {
            sum += y[*s];
        }

        let n_samples = samples[0].len();
        let mut all_false = vec![false; X.nrows()];

        self.node.split(
            X,
            y,
            samples,
            n_samples,
            vec![false; X.ncols()],
            &mut all_false,
            sum,
            &mut rng,
            0,
            &self.decision_tree_parameters,
        );
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Array1<f64> {
        let mut predictions = Array1::<f64>::zeros(X.nrows());
        for row in 0..X.nrows() {
            predictions[row] = self.predict_row(&X.row(row));
        }
        predictions
    }

    pub fn predict_row(&self, X: &ArrayView1<f64>) -> f64 {
        let mut node = &self.node;

        while let Some(feature_idx) = node.feature_index {
            if X[feature_idx] < node.feature_value.unwrap() {
                node = node.left_child.as_ref().unwrap();
            } else {
                node = node.right_child.as_ref().unwrap();
            }
        }
        node.label.unwrap()
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) {
        let samples: Vec<usize> = (0..X.nrows()).collect();
        self.fit_with_samples(X, y, &samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::load_iris;
    use ndarray::s;
    use rstest::*;

    #[rstest]
    fn test_tree() {
        let data = load_iris();
        let X = data.slice(s![.., 0..4]);
        let y = data.slice(s![.., 4]);

        let mut tree = DecisionTree::default();
        tree.fit(&X, &y);
        let predictions = tree.predict(&X);

        let mse = (&predictions - &y).mapv(|x| x * x).mean().unwrap();
        assert!(mse <= 0.02, "Got mse of {}.", mse);

        let mut another_tree = DecisionTree::default();
        another_tree.fit(&X, &predictions.view());
        let another_predictions = another_tree.predict(&X);

        // predictions were created by a decision tree. We should be able to
        // perfectly replicate these with another decision tree.
        assert_eq!(predictions - another_predictions, Array1::<f64>::zeros(150));
    }
}
