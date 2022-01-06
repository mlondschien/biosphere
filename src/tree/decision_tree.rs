use crate::tree::decision_tree_node::DecisionTreeNode;
use crate::tree::DecisionTreeParameters;
use crate::utils::argsort;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

static MIN_GAIN_TO_SPLIT: f64 = 1e-12;

pub struct DecisionTree {
    decision_tree_parameters: DecisionTreeParameters,
    node: DecisionTreeNode,
}

impl<'a> DecisionTree {
    pub fn new(decision_tree_parameters: DecisionTreeParameters) -> Self {
        DecisionTree {
            decision_tree_parameters,
            node: DecisionTreeNode::new(),
        }
    }

    pub fn default() -> Self {
        DecisionTree::new(DecisionTreeParameters::default())
    }

    pub fn fit_with_samples(
        &mut self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        samples: &[usize],
    ) {
        let sorted_samples = Vec::<&[usize]>::with_capacity(X.ncols());
        for idx in 0..X.ncols() {
            let sample_copy = samples.clone();
            sample_copy.sort_unstable_by(|a, b| X[[*a, idx]].partial_cmp(&X[[*b, idx]]).unwrap());
            sorted_samples.push(&sample_copy);
        }

        self.fit_with_sorted_samples(X, y, sorted_samples);
    }

    pub fn fit_with_sorted_samples(
        &mut self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        samples: Vec<&[usize]>,
    ) {
        let mut rng = StdRng::seed_from_u64(self.decision_tree_parameters.seed);

        let mut sum = 0.;
        for s in samples[0].iter() {
            sum += y[*s];
        }

        self.node.split(
            X,
            y,
            samples,
            vec![false; X.ncols()],
            sum,
            &mut rng,
            0,
            &self.decision_tree_parameters,
        )
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Array1<f64> {
        let mut predictions = Array1::<f64>::zeros(X.nrows());
        for row in 0..X.nrows() {
            predictions[row] = self.predict_row(&X.row(row));
        }
        predictions
    }

    pub fn predict_row(&self, X: &ArrayView1<f64>) -> f64 {
        let node = &self.node;

        while let Some(feature_idx) = node.feature_index {
            if X[feature_idx] < node.feature_value.unwrap() {
                node = &node.left_child.as_ref().unwrap();
            } else {
                node = &node.right_child.as_ref().unwrap();
            }
        }
        node.label.unwrap()
    }

    // pub fn fit(
    //     &mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>,
    // ) {
    //     let samples = (0..X.ncols()).map(|idx| argsort(&X.column(idx))).collect();
    //     self.fit_with_samples(X, y, samples)
    // }
}

// fn split_oob_samples<'b>(
//     oob_samples: &'b mut [usize],
//     X: &'_ ArrayView2<f64>,
//     best_feature: usize,
//     best_split_val: f64,
// ) -> (&'b mut [usize], &'b mut [usize]) {
//     let mut left_idx = 0;
//     let mut right_idx = oob_samples.len() - 1;

//     if right_idx > 0 {
//         'outer: loop {
//             while X[[oob_samples[left_idx], best_feature]] <= best_split_val {
//                 left_idx += 1;
//                 if left_idx == right_idx {
//                     break 'outer;
//                 }
//             }
//             while X[[oob_samples[right_idx], best_feature]] > best_split_val {
//                 right_idx -= 1;
//                 if left_idx == right_idx {
//                     break 'outer;
//                 }
//             }

//             oob_samples.swap(left_idx, right_idx);
//         }
//     }

//     if X[[oob_samples[left_idx], best_feature]] <= best_split_val {
//         left_idx += 1;
//     }

//     oob_samples.split_at_mut(left_idx)
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::testing::{is_sorted, load_iris};
//     use assert_approx_eq::*;
//     use ndarray::{arr1, arr2, s, Array, Array1};
//     use ndarray_rand::rand_distr::Uniform;
//     use ndarray_rand::RandomExt;
//     use rand::rngs::StdRng;
//     use rand::SeedableRng;
//     use rstest::*;

//     #[rstest]
//     #[case(&[0., 0., 0., 1., 1., 1.], 0, 6, 0, 3, 2.5, 0.25)]
//     #[case(&[0., 0., 0., 1., 1., 1.], 1, 5, 0, 3, 2.5, 0.25)]
//     #[case(&[0., 0., 0., 0., 0., 0.], 0, 6, 0, 0, 0., 0.)]
//     #[case(&[7., 1., 1., 1., 1., 1.], 0, 6, 0, 1, 0.5, 5.)]
//     #[case(&[7., 1., 1., 1., 1., 1.], 0, 2, 0, 1, 0.5, 9.)]
//     #[case(&[1., 1., 0., 0., 2., 2.], 0, 6, 0, 4, 3.5, 0.5)]
//     #[case(&[-5., -5., -5., -5., -5., 1.], 0, 6, 1, 5, 0.5, 5.)]
//     #[case(&[-5., -5., -5., -5., -5., 1.], 0, 6, 0, 5, 4.5, 5.)]
//     #[case(&[-5., 1., 1., 1., 1., 1., 1.], 0, 6, 0, 1, 0.5, 5.)]
//     #[case(&[-5., 1., 1., 1., 1., 1., 1.], 0, 6, 1, 5, 0.5, 0.2)]
//     fn test_find_best_split(
//         #[case] y: &[f64],
//         #[case] start: usize,
//         #[case] stop: usize,
//         #[case] feature: usize,
//         #[case] expected_split: usize,
//         #[case] expected_split_val: f64,
//         #[case] expected_gain: f64,
//     ) {
//         let X = arr2(&[[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 1.]]);
//         let X_view = X.view();
//         let y = arr1(y);
//         let y_view = y.view();

//         let tree = DecisionTree::default(&X_view, &y_view);

//         let (split, split_val, gain, _) =
//             tree.find_best_split(start, stop, feature, y.slice(s![start..stop]).sum());

//         assert_eq!((expected_split, expected_split_val), (split, split_val));

//         assert_approx_eq!(expected_gain, gain);
//     }

//     #[rstest]
//     #[case(0, 50, 100, 1)]
//     #[case(25, 50, 75, 1)]
//     #[case(25, 50, 100, 1)]
//     #[case(0, 6, 12, 1)]
//     fn test_split_samples(
//         #[case] start: usize,
//         #[case] split: usize,
//         #[case] stop: usize,
//         #[case] best_feature: usize,
//     ) {
//         let mut rng = StdRng::seed_from_u64(0);
//         let X = Array::random_using((100, 10), Uniform::new(0., 1.), &mut rng);
//         let X_view = X.view();
//         let y = Array::random_using(100, Uniform::new(0., 1.), &mut rng);
//         let y_view = y.view();

//         let all_false = vec![false; X.ncols()];
//         let mut tree = DecisionTree::default(&X_view, &y_view);

//         // Separate samples s.t. `tree.samples[feature_idx][start..stop]` contains the
//         // same indices for each `feature_idx`.
//         if start > 0 {
//             let x_sorted = X
//                 .column(best_feature)
//                 .select(Axis(0), &tree.samples[best_feature]);
//             let best_split_val = x_sorted[start] / 2. + x_sorted[start - 1] / 2.;
//             tree.split_samples(0, start, 100, &all_false, best_feature, best_split_val);
//         }

//         if stop < 100 {
//             let x_sorted = X
//                 .column(best_feature)
//                 .select(Axis(0), &tree.samples[best_feature]);
//             let best_split_val = x_sorted[stop] / 2. + x_sorted[stop - 1] / 2.;
//             tree.split_samples(start, stop, 100, &all_false, best_feature, best_split_val);
//         }

//         let x_sorted = X
//             .column(best_feature)
//             .select(Axis(0), &tree.samples[best_feature]);
//         let best_split_val = x_sorted[split] / 2. + x_sorted[split - 1] / 2.;

//         let samples_copy = tree.samples.clone();

//         tree.split_samples(start, split, stop, &all_false, best_feature, best_split_val);

//         for feature in 0..X.ncols() {
//             assert!(is_sorted(
//                 &X.column(feature)
//                     .select(Axis(0), &tree.samples[feature][start..split])
//             ));
//             assert!(is_sorted(
//                 &X.column(feature)
//                     .select(Axis(0), &tree.samples[feature][split..stop])
//             ));

//             for idx in tree.samples[feature][start..split].iter() {
//                 assert!(X[[*idx, best_feature]] <= best_split_val);
//             }

//             for idx in tree.samples[feature][split..stop].iter() {
//                 assert!(X[[*idx, best_feature]] > best_split_val);
//             }

//             let mut before = samples_copy[feature][start..stop].to_vec();
//             before.sort();
//             let mut after = tree.samples[feature][start..stop].to_vec();
//             after.sort();
//             assert_eq!(before, after);
//         }
//     }

//     #[rstest]
//     #[case(&mut [0, 1], &mut [0], &mut [1], 0, 0.5)]
//     #[case(&mut [0, 1], &mut [0, 1], &mut [], 0, 1.5)]
//     #[case(&mut [0, 1], &mut [], &mut [0, 1], 0, -1.)]
//     #[case(&mut [0, 1, 1, 2, 3], &mut [], &mut [0, 1, 1, 2, 3], 0, -1.)]
//     #[case(&mut [0, 1, 1, 2, 3], &mut [0, 1, 1, 2, 3], &mut [], 0, 10.)]
//     #[case(&mut [0, 3, 3, 2, 1], &mut [0, 1], &mut [3, 2, 3], 0, 1.5)]
//     #[case(&mut [0, 1, 2, 3, 4, 5], &mut [0, 1, 2, 3], &mut [4, 5], 1, 0.25)]
//     #[case(&mut [0, 2, 3, 0, 1, 4, 5], &mut [0, 2, 3, 0, 1], &mut [4, 5], 1, 0.25)]
//     fn test_split_oob_samples(
//         #[case] samples: &mut [usize],
//         #[case] expected_left: &mut [usize],
//         #[case] expected_right: &mut [usize],
//         #[case] best_feature: usize,
//         #[case] best_val: f64,
//     ) {
//         let X = arr2(&[
//             [0., 0.],
//             [1., -1.],
//             [2., 0.],
//             [3., -4.],
//             [4., 4.],
//             [5., 0.5],
//         ]);
//         let X_view = X.view();

//         let (left_samples, right_samples) =
//             split_oob_samples(samples, &X_view, best_feature, best_val);

//         assert_eq!(
//             (left_samples, right_samples),
//             (expected_left, expected_right)
//         );
//     }

//     #[rstest]
//     #[case(0, 100)]
//     #[case(50, 150)]
//     #[case(0, 150)]
//     fn test_tree_split(#[case] start: usize, #[case] stop: usize) {
//         let data = load_iris();
//         let X = data.slice(s![.., 0..4]);
//         let y = data.slice(s![.., 4]);

//         let mut oob_samples = (start..stop).collect::<Vec<_>>();

//         let mut tree = DecisionTree::default(&X, &y);
//         let mut rng = StdRng::seed_from_u64(0);
//         let result = tree.split(
//             0,
//             X.nrows(),
//             &mut oob_samples,
//             vec![false; 4],
//             0,
//             None,
//             &mut rng,
//         );

//         let mut predictions = Array1::zeros(stop - start);
//         for (idxs, val) in result.iter() {
//             for idx in idxs {
//                 predictions[idx - start] = *val;
//             }
//         }

//         let mse = (predictions - y.slice(s![start..stop]))
//             .mapv(|x| x * x)
//             .sum();
//         assert!(mse <= 2., "Got mse of {}.", mse);
//     }
// }
