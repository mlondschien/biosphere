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
    use crate::MaxFeatures;
    use ndarray::{s, Array2};
    use rstest::*;
    use std::collections::HashSet;

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

    #[test]
    fn test_fit_with_constant_features_predicts_mean() {
        let X = Array2::<f64>::zeros((10, 3));
        let y = Array1::from_vec(vec![0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]);

        let mut tree = DecisionTree::default();
        tree.fit(&X.view(), &y.view());
        let predictions = tree.predict(&X.view());

        assert_eq!(predictions, Array1::<f64>::from_elem(10, 0.5));
    }

    #[test]
    fn test_min_samples_leaf_prevents_split() {
        let X = Array2::from_shape_fn((10, 1), |(i, _)| i as f64);
        let y = Array1::from_vec(vec![0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]);

        // Tree should not SPLIT because min_samples_leaf of 6 would leave fewer than 6 samples in one of the child nodes!
        let parameters = DecisionTreeParameters::new(None, MaxFeatures::None, 2, 6, 0);
        let mut tree = DecisionTree::new(parameters);
        tree.fit(&X.view(), &y.view());
        let predictions = tree.predict(&X.view());

        assert_eq!(predictions, Array1::<f64>::from_elem(10, 0.5));
    }

    #[test]
    fn test_min_samples_leaf_allows_valid_split() {
        let X = Array2::from_shape_fn((4, 1), |(i, _)| i as f64);
        let y = Array1::from_vec(vec![0., 0., 1., 1.]);

        let parameters = DecisionTreeParameters::new(None, MaxFeatures::None, 2, 2, 0);
        let mut tree = DecisionTree::new(parameters);
        tree.fit(&X.view(), &y.view());
        let predictions = tree.predict(&X.view());

        assert_eq!(predictions, y);
    }

    #[test]
    fn test_constant_features_do_not_leak_across_sibling_nodes() {
        let mut X = Array2::<f64>::zeros((10, 2));
        for row in 5..10 {
            X[[row, 0]] = 1.;
        }
        for row in 7..10 {
            X[[row, 1]] = 1.;
        }
        let y = Array1::from_vec(vec![0., 0., 0., 0., 0., 9., 9., 11., 11., 11.]);

        let mut tree = DecisionTree::default();
        tree.fit(&X.view(), &y.view());
        let predictions = tree.predict(&X.view());

        assert_eq!(predictions, y);
    }

    #[test]
    fn test_predicts_three_classes_on_separable_data() {
        let n = 30;
        let X = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(n, |i| {
            if i < 10 {
                0.
            } else if i < 20 {
                1.
            } else {
                2.
            }
        });

        let mut tree = DecisionTree::default();
        tree.fit(&X.view(), &y.view());
        let predictions = tree.predict(&X.view());

        assert_eq!(predictions, y);
    }

    #[test]
    fn test_refit_overwrites_previous_tree() {
        let X = Array2::from_shape_fn((10, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(10, |i| if i < 5 { 0. } else { 10. });

        let mut tree = DecisionTree::default();
        tree.fit(&X.view(), &y.view());
        let predictions = tree.predict(&X.view());
        assert_eq!(predictions, y);

        let X_constant = Array2::<f64>::zeros((10, 1));
        let y_nonconstant = Array1::from_iter(1..=10).mapv(|x| x as f64);
        let expected = y_nonconstant.sum() / y_nonconstant.len() as f64;

        tree.fit(&X_constant.view(), &y_nonconstant.view());
        let predictions = tree.predict(&X_constant.view());

        assert_eq!(predictions, Array1::<f64>::from_elem(10, expected));
    }

    #[test]
    fn test_large_tree_multiclass_grid_predictions_are_stable() {
        let blocks_per_dim = 8usize;
        let samples_per_block_edge = 4usize;
        let grid_edge = blocks_per_dim * samples_per_block_edge;
        let n = grid_edge * grid_edge;

        let mut X = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);

        let mut idx = 0;
        for i in 0..grid_edge {
            for j in 0..grid_edge {
                X[[idx, 0]] = i as f64 + 0.5;
                X[[idx, 1]] = j as f64 + 0.5;

                let class_x = i / samples_per_block_edge;
                let class_y = j / samples_per_block_edge;
                y[idx] = (class_x + blocks_per_dim * class_y) as f64;

                idx += 1;
            }
        }

        let mut tree = DecisionTree::default();
        tree.fit(&X.view(), &y.view());

        // Fits the training grid exactly.
        assert_eq!(tree.predict(&X.view()), y);

        // Probes multiple points inside each class region (not part of training grid),
        // to ensure robust classification within each block.
        let probes_per_block = 4usize;
        let n_probes = blocks_per_dim * blocks_per_dim * probes_per_block;
        let mut X_probe = Array2::<f64>::zeros((n_probes, 2));
        let mut y_probe = Array1::<f64>::zeros(n_probes);

        let mut idx = 0;
        for class_y in 0..blocks_per_dim {
            for class_x in 0..blocks_per_dim {
                let base_x = (class_x * samples_per_block_edge) as f64;
                let base_y = (class_y * samples_per_block_edge) as f64;
                let class = (class_x + blocks_per_dim * class_y) as f64;

                let corners = [
                    (base_x + 0.1, base_y + 0.1),
                    (base_x + 0.1, base_y + 3.9),
                    (base_x + 3.9, base_y + 0.1),
                    (base_x + 3.9, base_y + 3.9),
                ];

                for (x0, x1) in corners {
                    X_probe[[idx, 0]] = x0;
                    X_probe[[idx, 1]] = x1;
                    y_probe[idx] = class;
                    idx += 1;
                }
            }
        }

        let y_pred = tree.predict(&X_probe.view()).mapv(|x| x.round());
        assert_eq!(y_pred, y_probe);

        let predicted_classes: HashSet<i32> = y_pred.iter().map(|x| *x as i32).collect();
        assert_eq!(predicted_classes.len(), blocks_per_dim * blocks_per_dim);
    }
}
