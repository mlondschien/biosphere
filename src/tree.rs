use ndarray::{Array1, Array2, Axis};

#[allow(dead_code)]
pub struct DecisionTree<'a> {
    X: &'a Array2<f64>,
    y: &'a Array1<f64>,
    max_depth: usize,
    features: Vec<usize>,
    /// For each feature contains `indices` s.t. `X[indices, feature]` is sorted.
    ordering: Vec<Vec<usize>>,
    // indices: Vec<usize>,
    // indices_oob: Vec<usize>,
}

impl<'a> DecisionTree<'a> {
    #[allow(dead_code)]
    /// Find best split for y[samples] at X[samples, feature].
    ///
    /// X[samples, feature] is assumed to be sorted.
    fn find_best_split(&self, feature: usize, samples: &[usize]) -> (usize, f64, f64) {
        let n = samples.len();

        let mut cumsum = self.y.select(Axis(0), samples);
        cumsum.accumulate_axis_inplace(Axis(0), |&prev, cur| *cur += prev);

        let mut max_gain = 0.;
        let mut gain: f64;
        let mut split = 0;

        let mut sum_times_s_by_n = 0.; // s * cumsum[n - 1] / n
        let sum_by_n = cumsum[n - 1] / n as f64; // cumsum[n - 1] / n

        for s in 1..(n - 1) {
            sum_times_s_by_n += sum_by_n;
            gain = (sum_times_s_by_n - cumsum[s - 1]).powi(2) / (s * (n - s)) as f64;

            if gain > max_gain {
                max_gain = gain;
                split = s;
            }
        }

        let split_val =
            self.X[[samples[split], feature]] / 2. + self.X[[samples[split - 1], feature]] / 2.;
        (split, split_val, max_gain)
    }

    #[allow(unused)]
    fn split(
        &self,
        samples: Vec<Vec<usize>>,
        oob_samples: &'a mut [usize],
        features: Vec<usize>,
        depth: usize,
    ) -> Vec<(&'a [usize], f64)> {
        if depth >= self.max_depth || samples[0].len() <= 2 {
            return vec![(oob_samples, self.mean(&samples[0]))];
        }

        let mut best_gain = 0.;
        let mut best_split = 0;
        let mut best_split_val = 0.;
        let mut best_feature = 0;

        for (feature_idx, feature) in self.features.iter().enumerate() {
            let (split, split_val, gain) = self.find_best_split(*feature, &samples[feature_idx]);

            if gain > best_gain {
                best_gain = gain;
                best_split = split;
                best_split_val = split_val;
                best_feature = *feature;
            }
        }

        if best_gain <= 0. {
            return vec![(oob_samples, self.mean(&samples[0]))];
        }

        let (left_samples, right_samples) =
            self.split_samples(samples, best_split, best_feature, best_split_val);
        let (mut left_oob_samples, mut right_oob_samples) =
            self.split_oob_samples(oob_samples, best_feature, best_split_val);

        let mut left = self.split(left_samples, left_oob_samples, features.clone(), depth + 1);
        let mut right = self.split(right_samples, right_oob_samples, features, depth + 1);

        // let left_in_bag_indices = in_bag_indices
        //     .iter()
        //     .filter(|&&idx| self.X[[idx, best_feature]] < best_split_val)
        //     .cloned()
        //     .collect();
        // let left_oob_indices = oob_indices
        //     .iter()
        //     .filter(|&&idx| self.X[[idx, best_feature]] < best_split_val)
        //     .cloned()
        //     .collect();
        // let right_in_bag_indices = in_bag_indices
        //     .iter()
        //     .filter(|&&idx| self.X[[idx, best_feature]] >= best_split_val)
        //     .cloned()
        //     .collect();
        // let right_oob_indices = oob_indices
        //     .iter()
        //     .filter(|&&idx| self.X[[idx, best_feature]] >= best_split_val)
        //     .cloned()
        //     .collect();

        // let mut right = self.split(right_in_bag_indices, right_oob_indices, depth + 1);
        // left.append(&mut right);
        left
    }

    /// Calculate mean value of y[samples].
    fn mean(&self, samples: &[usize]) -> f64 {
        let mut sum = 0.;
        for idx in samples {
            sum += self.y[*idx];
        }
        sum / samples.len() as f64
    }

    /// Split samples into two.
    ///
    /// Parameters:
    /// -----------
    /// samples:
    ///     For each `feature` in `self.features`, this should contain indices such that
    ///     `self.X[samples[feature_idx], feature]` is sorted.
    /// left_size:
    ///     Expected number of samples expected in `left_sample` output. Supplying this
    ///     allows efficient memory allocation.
    /// best_feature:
    ///     Feature by which obervations are split.
    /// best_split_val:
    ///     Value for `best_feature` at which observations are split.
    fn split_samples(
        &self,
        samples: Vec<Vec<usize>>,
        left_size: usize,
        best_feature: usize,
        best_split_val: f64,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let n_features = self.features.len();
        let n = samples[0].len(); // Unequal to X.shape()[0];
        let right_size = n - left_size;

        let mut left_samples = vec![Vec::<usize>::with_capacity(left_size); n_features];
        let mut right_samples = vec![Vec::<usize>::with_capacity(right_size); n_features];

        let mut sample: usize;

        for feature_idx in 0..n_features {
            let mut left_idx: usize = 0;
            let mut right_idx: usize = 0;

            for idx in 0..n {
                sample = samples[feature_idx][idx];
                if self.X[[sample, best_feature]] <= best_split_val {
                    left_samples[feature_idx][left_idx] = sample;
                    left_idx += 1;
                } else {
                    right_samples[feature_idx][right_idx] = sample;
                    right_idx += 1;
                }
            }
        }

        (left_samples, right_samples)
    }

    fn split_oob_samples(
        &self,
        oob_samples: &'a mut [usize],
        best_feature: usize,
        best_split_val: f64,
    ) -> (&'a mut [usize], &'a mut [usize]) {
        let mut left_idx = 0;
        let mut right_idx = oob_samples.len() - 1;

        while left_idx < right_idx {
            if self.X[[oob_samples[left_idx], best_feature]] <= best_split_val {
                left_idx += 1;
            } else {
                oob_samples.swap(left_idx, right_idx);
                right_idx -= 1;
            }
        }

        oob_samples.split_at_mut(left_idx)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::DecisionTree;
//     use ndarray::{arr1, arr2};
//     use rstest::*;

//     #[rstest]
//     #[case(2, 1.5, 0)]
//     #[case(3, 2.5, 1)]
//     fn test_find_best_split(
//         #[case] expected_split: usize,
//         #[case] expected_split_val: f64,
//         #[case] feature: usize,
//     ) {
//         let X = arr2(&[[0., 1., 2., 3., 4., 5.], [3., 3., 2., 1., 1., 4.]])
//             .t()
//             .to_owned();
//         let y = arr1(&[0., 0., 2., 1., 2., 2.]);
//         let tree = DecisionTree {
//             X: &X,
//             y: &y,
//             max_depth: 1,
//             features: vec![0, 1],
//             ordering: vec![vec![1, 1]; 1],
//         };
//         let (split, split_val, _) = tree.find_best_split(feature, &[0, 1, 2, 3, 4, 5]);

//         assert_eq!(expected_split, split);

//         assert_eq!(expected_split_val, split_val);
//     }

//     #[test]
//     fn test_split() {
//         let X = arr2(&[
//             [0., 0.],
//             [0., 2.],
//             [0., 4.],
//             [0., 5.],
//             [1., 1.],
//             [2., 2.],
//             [2., 3.],
//             [2., 4.],
//             [3., 1.],
//             [3., 5.],
//             [4., 0.],
//             [4., 2.],
//             [4., 3.],
//             [4., 4.],
//         ]);
//         let y = arr1(&[1., 1., 2., 2., 1., 1., 2., 2., 0., -1., 0., 0., -1., -1.]);

//         let in_bag_indices = vec![0, 1, 2, 5, 6, 7, 8, 9, 11, 12];
//         let oob_indices = vec![3, 4, 10, 13];

//         let tree = DecisionTree {
//             X: &X,
//             y: &y,
//             max_depth: 3,
//             features: vec![0, 1],
//             ordering: vec![vec![1, 1]; 1],
//         };

//         let result = tree.split(in_bag_indices, oob_indices, 0);

//         let mut predictions = vec![-7.; 14];
//         for (idxs, prediction) in result.iter() {
//             for idx in idxs {
//                 predictions[*idx] = *prediction;
//             }
//         }

//         assert_eq!(
//             predictions,
//             vec![-7., -7., -7., 2.0, 1.0, -7., -7., -7., -7., -7., 0., -7., -7., -1.,]
//         );
//     }
// }
