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
    fn split(
        &self,
        samples: Vec<Vec<usize>>,
        oob_samples: &'a mut [usize],
        features: Vec<usize>,
        depth: usize,
    ) -> Vec<(&'a [usize], f64)> {
        if depth >= self.max_depth || samples[0].len() <= 2 {
            return vec![(oob_samples, mean(self.y, &samples[0]))];
        }

        let mut best_gain = 0.;
        let mut best_split = 0;
        let mut best_split_val = 0.;
        let mut best_feature = 0;

        for (feature_idx, feature) in self.features.iter().enumerate() {
            let (split, split_val, gain) =
                find_best_split(self.X, self.y, *feature, &samples[feature_idx]);

            if gain > best_gain {
                best_gain = gain;
                best_split = split;
                best_split_val = split_val;
                best_feature = *feature;
            }
        }

        if best_gain <= 0. {
            return vec![(oob_samples, mean(self.y, &samples[0]))];
        }

        let (left_samples, right_samples) =
            split_samples(samples, best_split, self.X, best_feature, best_split_val);
        let (left_oob_samples, right_oob_samples) =
            split_oob_samples(oob_samples, self.X, best_feature, best_split_val);

        let mut left = self.split(left_samples, left_oob_samples, features.clone(), depth + 1);
        let mut right = self.split(right_samples, right_oob_samples, features, depth + 1);

        left.append(&mut right);
        left
    }
}

/// Find best split for y[samples] at X[samples, feature].
///
/// Parameters
/// ----------
/// X
///     2D array of shape (n_samples, n_features)
/// y
///     1D array of shape (n_samples)
/// feature
///     Index of feature for which to find the best split.
/// samples:
///     Indices of observations between which to find the best split. X[samples, feature]
///     should be sorted.
///
/// Returns
/// -------
/// split
///     Index of the best split. Left / right samples are samples[:split] and samples[split:].
/// split_val
///     Value at which to split. Equal to X[split, feature] / 2. + X[split - 1, feature].
/// gain
///     Gain when splitting at split (or split_val).
// TODO: Disallow split points for which X[split, feature] == X[split - 1, feature].
fn find_best_split(
    X: &Array2<f64>,
    y: &Array1<f64>,
    feature: usize,
    samples: &[usize],
) -> (usize, f64, f64) {
    let n = samples.len();

    let mut cumsum = y.select(Axis(0), samples);
    cumsum.accumulate_axis_inplace(Axis(0), |&prev, cur| *cur += prev);

    println!("{:?}", cumsum);
    let mut max_gain = 0.;
    let mut gain: f64;
    let mut split = 0;

    let mut sum_times_s_by_n = 0.; // s * cumsum[n - 1] / n
    let sum_by_n = cumsum[n - 1] / n as f64; // cumsum[n - 1] / n

    for s in 1..n {
        sum_times_s_by_n += sum_by_n;
        gain = (sum_times_s_by_n - cumsum[s - 1]).powi(2) / (s * (n - s)) as f64;
        println!("s:{}, gain:{:?}", s, gain);
        if gain > max_gain {
            max_gain = gain;
            split = s;
        }
    }
    let split_val: f64;

    // TODO: The case split=0 is irrelevant. Get rid of the if/else.
    if split != 0 {
        split_val = X[[samples[split], feature]] / 2. + X[[samples[split - 1], feature]] / 2.;
    } else {
        split_val = X[[samples[0], feature]];
    }

    (split, split_val, max_gain)
}

/// Calculate mean value of y[samples].
fn mean(y: &Array1<f64>, samples: &[usize]) -> f64 {
    let mut sum = 0.;
    for idx in samples {
        sum += y[*idx];
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
///
/// Returns:
/// --------
/// left_sample:
///  
/// right_sample:
///
// TODO: For the feature which was used to split, this is trivial / can be sped up.
fn split_samples(
    samples: Vec<Vec<usize>>,
    left_size: usize,
    X: &Array2<f64>,
    best_feature: usize,
    best_split_val: f64,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n_features = samples.len();
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
            if X[[sample, best_feature]] <= best_split_val {
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

fn split_oob_samples<'a>(
    oob_samples: &'a mut [usize],
    X: &'_ Array2<f64>,
    best_feature: usize,
    best_split_val: f64,
) -> (&'a mut [usize], &'a mut [usize]) {
    let mut left_idx = 0;
    let mut right_idx = oob_samples.len() - 1;

    while left_idx < right_idx {
        if X[[oob_samples[left_idx], best_feature]] <= best_split_val {
            left_idx += 1;
        } else {
            oob_samples.swap(left_idx, right_idx);
            right_idx -= 1;
        }
    }

    oob_samples.split_at_mut(left_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::is_sorted;
    use assert_approx_eq::*;
    use ndarray::{arr1, arr2, s};
    use rstest::*;

    #[rstest]
    #[case(&[0., 0., 0., 1., 1., 1.], &[0, 1, 2, 3, 4, 5], 0, 3, 2.5, 0.25)]
    #[case(&[0., 0., 0., 1., 1., 1.], &[0, 2, 4, 5], 0, 2, 3., 0.25)]
    #[case(&[0., 0., 0., 1., 1., 1.], &[0, 0, 2, 3, 4, 5], 0, 3, 2.5, 0.25)]
    #[case(&[0., 0., 0., 1., 1., 1.], &[0, 0, 1, 2, 2, 3, 4, 4, 4, 5], 0, 5, 2.5, 0.25)]
    //
    #[case(&[0., 0., 0., 0., 0., 0.], &[0, 1, 2, 3, 4, 5], 0, 0, 0., 0.)]
    //
    #[case(&[7., 1., 1., 1., 1., 1.], &[0, 1, 2, 3, 4, 5], 0, 1, 0.5, 5.)]
    #[case(&[1., 1., 0., 0., 2., 2.], &[0, 1, 2, 3, 4, 5], 0, 4, 3.5, 0.5)]
    // //
    #[case(&[-5., -5., -5., -5., -5., 1.], &[0, 1, 2, 3, 4, 5], 1, 5, 0.5, 5.)]
    #[case(&[-5., -5., -5., -5., -5., 1.], &[4, 3, 3, 1, 4, 5], 1, 5, 0.5, 5.)]
    // TODO #[case(&[0., 1., 1., 1., 1., 1., 1.], &[0, 1, 2, 3, 4, 5], 1, 5, 0.5, 5.)]
    fn test_find_best_split(
        #[case] y: &[f64],
        #[case] samples: &[usize],
        #[case] feature: usize,
        #[case] expected_split: usize,
        #[case] expected_split_val: f64,
        #[case] expected_gain: f64,
    ) {
        let X = arr2(&[[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 1.]]);
        let y = arr1(y);

        assert!(is_sorted(
            &X.slice(s![.., feature])
                .select(Axis(0), samples)
                .as_slice()
                .unwrap()
        ));

        let (split, split_val, gain) = find_best_split(&X, &y, feature, samples);

        assert_eq!(
            (expected_split, expected_split_val, expected_gain),
            (split, split_val, gain)
        );
    }

    // #[test]
    // fn test_split() {
    //     let X = arr2(&[
    //         [0., 0.],
    //         [0., 2.],
    //         [0., 4.],
    //         [0., 5.],
    //         [1., 1.],
    //         [2., 2.],
    //         [2., 3.],
    //         [2., 4.],
    //         [3., 1.],
    //         [3., 5.],
    //         [4., 0.],
    //         [4., 2.],
    //         [4., 3.],
    //         [4., 4.],
    //     ]);
    //     let y = arr1(&[1., 1., 2., 2., 1., 1., 2., 2., 0., -1., 0., 0., -1., -1.]);

    //     let in_bag_indices = vec![0, 1, 2, 5, 6, 7, 8, 9, 11, 12];
    //     let oob_indices = vec![3, 4, 10, 13];

    //     let tree = DecisionTree {
    //         X: &X,
    //         y: &y,
    //         max_depth: 3,
    //         features: vec![0, 1],
    //         ordering: vec![vec![1, 1]; 1],
    //     };

    //     let result = tree.split(in_bag_indices, oob_indices, 0);

    //     let mut predictions = vec![-7.; 14];
    //     for (idxs, prediction) in result.iter() {
    //         for idx in idxs {
    //             predictions[*idx] = *prediction;
    //         }
    //     }

    //     assert_eq!(
    //         predictions,
    //         vec![-7., -7., -7., 2.0, 1.0, -7., -7., -7., -7., -7., 0., -7., -7., -1.,]
    //     );
    // }
}
