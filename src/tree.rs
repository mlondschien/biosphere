use crate::utils::argsort;
use ndarray::{ArrayView1, ArrayView2, Axis};

pub struct DecisionTree<'a> {
    pub X: &'a ArrayView2<'a, f64>,
    pub y: &'a ArrayView1<'a, f64>,
    // Vector of vectors of in-bag indices. The outer vector should be of the same
    // length as features. All inner vectors should be of the same length, each
    // containing the same in-bag indices, however in different order. The ordering
    // should be such that for each `idx` in `0..features.len()`, the array
    // `X[samples[idx], features[idx]]` is ordered. `samples` will be reshuffled during
    // `fit`.
    samples: Vec<Vec<usize>>,
    // Subset of 0..X.ncols(). Only columns corresponding to an entry `features` are
    // used to split. Useful for random forests.
    features: Vec<usize>,
    // Maximum depth of the tree.
    pub max_depth: Option<u16>,
    // Minimum number of samples required to split a node.
    pub min_samples_split: usize,
    // Minimum gain required to split a node. Can be seen as a threshold.
    pub min_gain_to_split: f64,
}

impl<'a> DecisionTree<'a> {
    pub fn new(
        X: &'a ArrayView2<'a, f64>,
        y: &'a ArrayView1<'a, f64>,
        features: Vec<usize>,
        samples: Vec<Vec<usize>>,
        max_depth: Option<u16>,
        min_samples_split: Option<usize>,
        min_gain_to_split: Option<f64>,
    ) -> Self {
        DecisionTree {
            X,
            y,
            features,
            samples,
            max_depth,
            min_samples_split: min_samples_split.unwrap_or(2),
            min_gain_to_split: min_gain_to_split.unwrap_or(1e-6),
        }
    }

    pub fn default(X: &'a ArrayView2<'a, f64>, y: &'a ArrayView1<'a, f64>) -> Self {
        let samples = X
            .axis_iter(Axis(1))
            .map(|x| argsort(&x))
            .collect::<Vec<Vec<usize>>>();
        DecisionTree::new(X, y, (0..X.ncols()).collect(), samples, None, None, None)
    }

    pub fn split(
        &mut self,
        // For each `idx` in `feature_indices`, `self.samples[idx][start..stop]` are the
        // in-bag samples of the current node. Furthermore, these should be such that
        // `X[self.samples[idx][start..stop], self.features[idx]]` is ordered.
        start: usize,
        stop: usize,
        oob_samples: &mut [usize],
        // Subset of 0..`self.features.len()`. Only columns `self.features[idx]` for
        // `idx` in `feature_indices` are used to split. When a column with index
        // `self.features[idx]` is constant, it is removed from `feature_indices`.
        // Note that entries in `feature_indices` do not refer directly to columns in
        // `self.X`, but rather to entries in `self.features`.
        mut feature_indices: Vec<usize>,
        current_depth: u16,
    ) -> Vec<(Vec<usize>, f64)> {
        if oob_samples.is_empty() {
            return vec![];
        }

        if (self.max_depth.is_some() && current_depth >= self.max_depth.unwrap())
            || (stop - start) <= self.min_samples_split
        {
            return vec![(oob_samples.to_vec(), self.mean(start, stop))];
        }

        let mut best_gain = 0.;
        let mut best_split = 0;
        let mut best_split_val = 0.;
        let mut best_feature_idx = 0;

        let mut feature_indices_to_remove: Vec<usize> = vec![];

        for &feature_idx in feature_indices.iter() {
            let (split, split_val, gain) = self.find_best_split(start, stop, feature_idx);

            if gain < self.min_gain_to_split {
                // feature self.features[feature_idx] appears to be constant at this
                // node. We'll remove it from the list of features afterwards.
                feature_indices_to_remove.push(feature_idx);
            } else if gain > best_gain {
                best_gain = gain;
                best_split = split;
                best_split_val = split_val;
                best_feature_idx = feature_idx;
            }
        }

        if best_gain <= self.min_gain_to_split {
            return vec![(oob_samples.to_vec(), self.mean(start, stop))];
        }

        if !feature_indices_to_remove.is_empty() {
            feature_indices.retain(|x| !feature_indices_to_remove.contains(x));
        }

        self.split_samples(
            start,
            best_split,
            stop,
            &feature_indices,
            best_feature_idx,
            best_split_val,
        );

        let (left_oob_samples, right_oob_samples) = split_oob_samples(
            oob_samples,
            self.X,
            self.features[best_feature_idx],
            best_split_val,
        );

        let mut left = self.split(
            start,
            best_split,
            left_oob_samples,
            feature_indices.clone(),
            current_depth + 1,
        );
        let mut right = self.split(
            best_split,
            stop,
            right_oob_samples,
            feature_indices,
            current_depth + 1,
        );
        left.append(&mut right);
        left
    }

    /// Calculate mean value of y[samples[0][start..stop]].
    fn mean(&self, start: usize, stop: usize) -> f64 {
        let mut sum = 0.;
        for idx in self.samples[0][start..stop].iter() {
            sum += self.y[*idx];
        }
        sum / (stop - start) as f64
    }

    /// Find the best split in `self.X[start..stop, self.features[feature_idx]`.
    fn find_best_split(&self, start: usize, stop: usize, feature_idx: usize) -> (usize, f64, f64) {
        let feature = self.features[feature_idx];
        let samples = &self.samples[feature_idx];
        // X is constant in this segment.
        if self.X[[samples[stop - 1], feature]] - self.X[[samples[start], feature]] < 1e-6 {
            return (0, 0., 0.);
        }

        let mut cumsum = self.y.select(Axis(0), &samples[start..stop]);
        cumsum.accumulate_axis_inplace(Axis(0), |&prev, cur| *cur += prev);

        let n = stop - start;
        let mut max_gain = 0.;
        let mut gain: f64;
        let mut split = start;

        let mut sum_times_s_by_n = 0.; // s * cumsum[n - 1] / n
        let sum_by_n = cumsum[n - 1] / n as f64; // cumsum[n - 1] / n

        for s in 1..n {
            sum_times_s_by_n += sum_by_n;

            // Hackedy hack.
            if self.X[[samples[s + start], feature]] - self.X[[samples[s + start - 1], feature]]
                < 1e-12
            {
                continue;
            }

            gain = (sum_times_s_by_n - cumsum[s - 1]).powi(2) / (s * (n - s)) as f64;
            if gain > max_gain {
                max_gain = gain;
                split = s;
            }
        }
        let split_val: f64;
        // println!(
        //     "Found best split, start: {}, stop: {}, feature {}, best_split: {}, max_gain: {}",
        //     start, stop, feature, split, max_gain
        // );

        if split == start {
            (0, 0., 0.)
        } else {
            split_val = self.X[[samples[split + start], feature]] / 2.
                + self.X[[samples[split + start - 1], feature]] / 2.;
            (split + start, split_val, max_gain)
        }
    }

    /// For each idx in `feature_indices`, reorder `samples[idx][start..stop]` such that
    /// indices `samples[idx][start..split]` point to observations that belong to the
    /// left node (i.e. have `x[best_feature] <= best_split_val`) and indices
    /// `samples[idx][split..stop]` point to observations that belong to the right node,
    /// while preserving that `self.X[start..split, samples[idx][start..split]` and
    /// `self.X[split..stop, samples[idx][split..stop]]` are sorted.
    fn split_samples(
        &mut self,
        start: usize,
        split: usize,
        stop: usize,
        feature_indices: &[usize],
        best_feature_idx: usize,
        best_split_val: f64,
    ) {
        let mut left_temp = Vec::<usize>::with_capacity(split - start);
        let mut right_temp = Vec::<usize>::with_capacity(stop - split);

        let mut samples: &[usize];
        let best_feature: usize = self.features[best_feature_idx];

        for &feature_idx in feature_indices {
            if feature_idx == best_feature_idx {
                continue;
            }
            samples = &self.samples[feature_idx];

            for &sample in samples[start..stop].iter() {
                if self.X[[sample, best_feature]] > best_split_val {
                    right_temp.push(sample);
                } else {
                    left_temp.push(sample)
                }
            }
            self.samples[feature_idx][split..stop].copy_from_slice(&right_temp);
            self.samples[feature_idx][start..split].copy_from_slice(&left_temp);
            left_temp.clear();
            right_temp.clear();
        }
    }
}

fn split_oob_samples<'b>(
    oob_samples: &'b mut [usize],
    X: &'_ ArrayView2<f64>,
    best_feature: usize,
    best_split_val: f64,
) -> (&'b mut [usize], &'b mut [usize]) {
    let mut left_idx = 0;
    let mut right_idx = oob_samples.len() - 1;

    if right_idx > 0 {
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
    }

    if X[[oob_samples[left_idx], best_feature]] <= best_split_val {
        left_idx += 1;
    }

    oob_samples.split_at_mut(left_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{is_sorted, load_iris};
    use ndarray::{arr1, arr2, s, Array, Array1};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rstest::*;

    #[rstest]
    #[case(&[0., 0., 0., 1., 1., 1.], 0, 6, 0, 3, 2.5, 0.25)]
    #[case(&[0., 0., 0., 1., 1., 1.], 1, 5, 0, 3, 2.5, 0.25)]
    #[case(&[0., 0., 0., 0., 0., 0.], 0, 6, 0, 0, 0., 0.)]
    #[case(&[7., 1., 1., 1., 1., 1.], 0, 6, 0, 1, 0.5, 5.)]
    #[case(&[7., 1., 1., 1., 1., 1.], 0, 2, 0, 1, 0.5, 9.)]
    #[case(&[1., 1., 0., 0., 2., 2.], 0, 6, 0, 4, 3.5, 0.5)]
    #[case(&[-5., -5., -5., -5., -5., 1.], 0, 6, 1, 5, 0.5, 5.)]
    #[case(&[-5., -5., -5., -5., -5., 1.], 0, 6, 0, 5, 4.5, 5.)]
    #[case(&[-5., 1., 1., 1., 1., 1., 1.], 0, 6, 0, 1, 0.5, 5.)]
    #[case(&[-5., 1., 1., 1., 1., 1., 1.], 0, 6, 1, 5, 0.5, 0.2)]
    fn test_find_best_split(
        #[case] y: &[f64],
        #[case] start: usize,
        #[case] stop: usize,
        #[case] feature: usize,
        #[case] expected_split: usize,
        #[case] expected_split_val: f64,
        #[case] expected_gain: f64,
    ) {
        let X = arr2(&[[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 1.]]);
        let X_view = X.view();
        let y = arr1(y);
        let y_view = y.view();

        let tree = DecisionTree::default(&X_view, &y_view);

        let (split, split_val, gain) = tree.find_best_split(start, stop, feature);

        assert_eq!(
            (expected_split, expected_split_val, expected_gain),
            (split, split_val, gain)
        );
    }

    #[rstest]
    #[case(0, 50, 100, 1)]
    #[case(25, 50, 75, 1)]
    #[case(25, 50, 100, 1)]
    #[case(0, 6, 12, 1)]
    fn test_split_samples(
        #[case] start: usize,
        #[case] split: usize,
        #[case] stop: usize,
        #[case] best_feature_idx: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(0);
        let X = Array::random_using((100, 10), Uniform::new(0., 1.), &mut rng);
        let X_view = X.view();
        let y = Array::random_using(100, Uniform::new(0., 1.), &mut rng);
        let y_view = y.view();

        let features = (0..X.shape()[1]).collect::<Vec<_>>();
        let mut tree = DecisionTree::default(&X_view, &y_view);

        // Separate samples s.t. `tree.samples[feature_idx][start..stop]` contains the
        // same indices for each `feature_idx`.
        if start > 0 {
            let x_sorted = X
                .column(best_feature_idx)
                .select(Axis(0), &tree.samples[best_feature_idx]);
            let best_split_val = x_sorted[start] / 2. + x_sorted[start - 1] / 2.;
            tree.split_samples(0, start, 100, &features, best_feature_idx, best_split_val);
        }

        if stop < 100 {
            let x_sorted = X
                .column(best_feature_idx)
                .select(Axis(0), &tree.samples[best_feature_idx]);
            let best_split_val = x_sorted[stop] / 2. + x_sorted[stop - 1] / 2.;
            tree.split_samples(
                start,
                stop,
                100,
                &features,
                best_feature_idx,
                best_split_val,
            );
        }

        let x_sorted = X
            .column(best_feature_idx)
            .select(Axis(0), &tree.samples[best_feature_idx]);
        let best_split_val = x_sorted[split] / 2. + x_sorted[split - 1] / 2.;

        let samples_copy = tree.samples.clone();

        tree.split_samples(
            start,
            split,
            stop,
            &features,
            best_feature_idx,
            best_split_val,
        );

        for feature_idx in features {
            assert!(is_sorted(
                &X.column(feature_idx)
                    .select(Axis(0), &tree.samples[feature_idx][start..split])
            ));
            assert!(is_sorted(
                &X.column(feature_idx)
                    .select(Axis(0), &tree.samples[feature_idx][split..stop])
            ));

            for idx in tree.samples[feature_idx][start..split].iter() {
                assert!(X[[*idx, best_feature_idx]] <= best_split_val);
            }

            for idx in tree.samples[feature_idx][split..stop].iter() {
                assert!(X[[*idx, best_feature_idx]] > best_split_val);
            }

            let mut before = samples_copy[feature_idx][start..stop].to_vec();
            before.sort();
            let mut after = tree.samples[feature_idx][start..stop].to_vec();
            after.sort();
            assert_eq!(before, after);
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
        let data = load_iris();
        let X = data.slice(s![.., 0..4]);
        let y = data.slice(s![.., 4]);

        let mut oob_samples = (start..stop).collect::<Vec<_>>();

        let mut tree = DecisionTree::default(&X, &y);
        let result = tree.split(0, X.nrows(), &mut oob_samples, vec![0, 1, 2, 3], 0);

        let mut predictions = Array1::zeros(stop - start);
        for (idxs, val) in result.iter() {
            for idx in idxs {
                predictions[idx - start] = *val;
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
