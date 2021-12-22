use ndarray::{ArrayView1, ArrayView2, Axis};

#[allow(dead_code)]
pub struct DecisionTree<'a> {
    pub X: &'a ArrayView2<'a, f64>,
    pub y: &'a ArrayView1<'a, f64>,
    pub max_depth: usize,
}

impl<'a> DecisionTree<'a> {
    #[allow(dead_code)]
    pub fn split(
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

        for (feature_idx, feature) in features.iter().enumerate() {
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
    X: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    feature: usize,
    samples: &[usize],
) -> (usize, f64, f64) {
    let n = samples.len();

    let mut cumsum = y.select(Axis(0), samples);
    cumsum.accumulate_axis_inplace(Axis(0), |&prev, cur| *cur += prev);

    let mut max_gain = 0.;
    let mut gain: f64;
    let mut split = 0;

    let mut sum_times_s_by_n = 0.; // s * cumsum[n - 1] / n
    let sum_by_n = cumsum[n - 1] / n as f64; // cumsum[n - 1] / n

    for s in 1..n {
        sum_times_s_by_n += sum_by_n;
        gain = (sum_times_s_by_n - cumsum[s - 1]).powi(2) / (s * (n - s)) as f64;
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
fn mean(y: &ArrayView1<f64>, samples: &[usize]) -> f64 {
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
///     For each `feature` in `features`, this should contain indices such that
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
    X: &ArrayView2<f64>,
    best_feature: usize,
    best_split_val: f64,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n_features = samples.len();
    let n_samples = samples[0].len();
    let right_size = n_samples - left_size;

    let mut left_samples = vec![Vec::<usize>::with_capacity(left_size); n_features];
    let mut right_samples = vec![Vec::<usize>::with_capacity(right_size); n_features];

    let mut sample: usize;
    for feature_idx in 0..n_features {
        for idx in 0..n_samples {
            sample = samples[feature_idx][idx];
            if X[[sample, best_feature]] <= best_split_val {
                left_samples[feature_idx].push(sample);
            } else {
                right_samples[feature_idx].push(sample);
            }
        }
    }

    (left_samples, right_samples)
}

fn split_oob_samples<'a>(
    oob_samples: &'a mut [usize],
    X: &'_ ArrayView2<f64>,
    best_feature: usize,
    best_split_val: f64,
) -> (&'a mut [usize], &'a mut [usize]) {
    let mut left_idx = 0;
    let mut right_idx = oob_samples.len() - 1;

    'outer: loop {
        while X[[oob_samples[left_idx], best_feature]] <= best_split_val {
            left_idx += 1;
            if left_idx == right_idx {
                break 'outer;
            }
        }
        while X[[oob_samples[right_idx], best_feature]] > best_split_val {
            right_idx -= 1;
            if left_idx == right_idx {
                break 'outer;
            }
        }

        oob_samples.swap(left_idx, right_idx);
    }

    if X[[oob_samples[left_idx], best_feature]] <= best_split_val {
        left_idx += 1;
    }

    oob_samples.split_at_mut(left_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{arrange_samples, is_sorted};
    use assert_approx_eq::*;
    use csv::ReaderBuilder;
    use ndarray::{arr1, arr2, s, Array1, Array2};
    use ndarray_csv::Array2Reader;
    use rstest::*;
    use std::fs::File;

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
        let X_view = X.view();
        let y = arr1(y);
        let y_view = y.view();

        assert!(is_sorted(
            &X.slice(s![.., feature])
                .select(Axis(0), samples)
                .as_slice()
                .unwrap()
        ));

        let (split, split_val, gain) = find_best_split(&X_view, &y_view, feature, samples);

        assert_eq!(
            (expected_split, expected_split_val, expected_gain),
            (split, split_val, gain)
        );
    }

    #[rstest]
    #[case(&[0., 0., 0., 1., 1., 1.], &[0, 1, 2, 3, 4, 5], 0.5)]
    #[case(&[0., 0., 0., 1., 1., 1.], &[0], 0.)]
    #[case(&[0., 0., 0., 1., 1., 1.], &[0, 3], 0.5)]
    #[case(&[-1., 0., 1., 2., 3., 4.], &[0, 0, 1, 5], 0.5)]
    fn test_mean(#[case] y: &[f64], #[case] samples: &[usize], #[case] expected_mean: f64) {
        let y = arr1(y);
        let y_view = y.view();
        assert_approx_eq!(expected_mean, mean(&y_view, samples));
    }

    #[rstest]
    #[case(&[0, 1, 2, 3, 4, 5], 0, 2.5)]
    #[case(&[0, 1, 1, 2, 5], 0, 2.5)]
    #[case(&[0, 0, 0, 1, 1, 2, 4, 4, 4, 4, 5], 0, 2.5)]
    //
    #[case(&[0, 1, 2, 3, 4, 5], 0, 0.5)]
    #[case(&[0, 1, 2, 3, 4, 5], 0, 0.)]
    #[case(&[0, 1, 2, 3, 4, 5], 0, -1.)]
    #[case(&[0, 1, 2, 3, 4, 5], 0, 5.)]
    //
    #[case(&[0, 1, 2, 3, 4, 5], 1, 0.25)]
    #[case(&[0, 0, 0, 0, 2, 3, 4, 5], 1, 0.25)]
    #[case(&[0, 0, 0, 0, 2, 3, 4, 5], 1, 0.75)]
    fn test_split_samples(
        #[case] sample_counts: &[usize],
        #[case] best_feature: usize,
        #[case] best_split_val: f64,
    ) {
        let X = arr2(&[
            [0., 0.],
            [1., -1.],
            [2., 0.],
            [3., -4.],
            [4., 4.],
            [5., 0.5],
        ]);
        let X_view = X.view();
        let features = (0..X.shape()[1]).collect::<Vec<_>>();

        let samples = arrange_samples(sample_counts, &features, &X_view);

        // left_size is only used for efficient memory allocation.
        let (left_samples, right_samples) =
            split_samples(samples, 0, &X_view, best_feature, best_split_val);

        for feature_idx in features {
            assert!(is_sorted(
                X.slice(s![.., feature_idx])
                    .select(Axis(0), &left_samples[feature_idx])
                    .as_slice()
                    .unwrap()
            ));
            assert!(is_sorted(
                X.slice(s![.., feature_idx])
                    .select(Axis(0), &right_samples[feature_idx])
                    .as_slice()
                    .unwrap()
            ));

            for idx in left_samples[feature_idx].iter() {
                assert!(X[[*idx, best_feature]] <= best_split_val);
            }

            for idx in right_samples[feature_idx].iter() {
                assert!(X[[*idx, best_feature]] > best_split_val);
            }

            // Assert set(left_sample[feature_idx]) + set(right_sample[feature_idx]) == set(sample_counts)
            let mut full_sample = left_samples[feature_idx].clone();
            full_sample.append(&mut right_samples[feature_idx].clone());
            full_sample.sort();
            assert_eq!(full_sample, sample_counts);
        }
    }

    #[rstest]
    #[case(&mut [0, 1], &mut [0], &mut [1], 0, 0.5)]
    #[case(&mut [0, 1], &mut [0, 1], &mut [], 0, 1.5)]
    #[case(&mut [0, 1], &mut [], &mut [0, 1], 0, -1.)]
    #[case(&mut [0, 1, 1, 2, 3], &mut [], &mut [0, 1, 1, 2, 3], 0, -1.)]
    #[case(&mut [0, 1, 1, 2, 3], &mut [0, 1, 1, 2, 3], &mut [], 0, 10.)]
    #[case(&mut [0, 3, 3, 2, 1], &mut [0, 1], &mut [3, 2, 3], 0, 1.5)]
    #[case(&mut [0, 1, 2, 3, 4, 5], &mut [0, 1, 2, 3], &mut [4, 5], 1, 0.25)]
    #[case(&mut [0, 2, 3, 0, 1, 4, 5], &mut [0, 2, 3, 0, 1], &mut [4, 5], 1, 0.25)]
    fn test_split_oob_samples(
        #[case] samples: &mut [usize],
        #[case] expected_left: &mut [usize],
        #[case] expected_right: &mut [usize],
        #[case] best_feature: usize,
        #[case] best_val: f64,
    ) {
        let X = arr2(&[
            [0., 0.],
            [1., -1.],
            [2., 0.],
            [3., -4.],
            [4., 4.],
            [5., 0.5],
        ]);
        let X_view = X.view();

        let (left_samples, right_samples) =
            split_oob_samples(samples, &X_view, best_feature, best_val);

        assert_eq!(
            (left_samples, right_samples),
            (expected_left, expected_right)
        );
    }

    #[rstest]
    #[case(0, 100)]
    #[case(50, 150)]
    #[case(0, 150)]
    fn test_tree_split(#[case] start: usize, #[case] stop: usize) {
        let file = File::open("testdata/iris.csv").unwrap();
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
        let data: Array2<f64> = reader.deserialize_array2((150, 5)).unwrap();

        let X = data.slice(s![.., 0..4]);
        let y = data.slice(s![.., 4]);

        let mut oob_samples = (start..stop).collect::<Vec<_>>();
        let samples = arrange_samples(&oob_samples, &[0, 1, 2, 3], &X);

        let tree = DecisionTree {
            X: &X,
            y: &y,
            max_depth: 8,
        };
        let result = tree.split(samples, &mut oob_samples, vec![0, 1, 2, 3], 0);

        let mut predictions = Array1::zeros(stop - start);
        for (idxs, val) in result.iter() {
            for idx in *idxs {
                predictions[*idx - start] = *val;
            }
        }

        assert!(
            (predictions - y.slice(s![start..stop]))
                .mapv(|x| x * x)
                .sum()
                < 1.
        );
    }
}
