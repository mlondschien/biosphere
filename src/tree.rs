use crate::utils::argsort;
use ndarray::{ArrayView1, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::Rng;

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
    // Maximum depth of the tree.
    pub max_depth: Option<u16>,
    pub mtry: u16,
    // Minimum number of samples required to split a node.
    pub min_samples_split: usize,
    // Minimum gain required to split a node. Can be seen as a threshold.
    pub min_gain_to_split: f64,
    //
    pub min_samples_leaf: usize,
}

impl<'a> DecisionTree<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        X: &'a ArrayView2<'a, f64>,
        y: &'a ArrayView1<'a, f64>,
        samples: Vec<Vec<usize>>,
        max_depth: Option<u16>,
        mtry: u16,
        min_samples_split: Option<usize>,
        min_gain_to_split: Option<f64>,
        min_samples_leaf: Option<usize>,
    ) -> Self {
        DecisionTree {
            X,
            y,
            samples,
            max_depth,
            mtry,
            min_samples_split: min_samples_split.unwrap_or(2),
            min_gain_to_split: min_gain_to_split.unwrap_or(1e-12),
            min_samples_leaf: min_samples_leaf.unwrap_or(1),
        }
    }

    pub fn default(X: &'a ArrayView2<'a, f64>, y: &'a ArrayView1<'a, f64>) -> Self {
        let samples = X
            .axis_iter(Axis(1))
            .map(|x| argsort(&x))
            .collect::<Vec<Vec<usize>>>();
        DecisionTree::new(X, y, samples, None, X.ncols() as u16, None, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split(
        &mut self,
        // For each `idx` in `feature_indices`, `self.samples[idx][start..stop]` are the
        // in-bag samples of the current node. Furthermore, these should be such that
        // `X[self.samples[idx][start..stop], self.features[idx]]` is ordered.
        start: usize,
        stop: usize,
        oob_samples: &mut [usize],
        // Vector of length `self.X.ncols()`. Initially values are all false. When one
        // feature is observed to be constant at a node, the corresponding entry in
        // `constant_features` is set to true and passed to children. We can then
        // avoid the expensive computation of the maximal gain for that feature at the
        // node.
        mut constant_features: Vec<bool>,
        current_depth: u16,
        // sum over y[samples[idx]] for idx in start..stop.
        sum: Option<f64>,
        rng: &mut impl Rng,
    ) -> Vec<(Vec<usize>, f64)> {
        assert!(start < stop);

        if oob_samples.is_empty() {
            return vec![];
        }

        let sum = sum.unwrap_or_else(|| self.sum(start, stop));
        let mean = sum / (stop - start) as f64;

        if (self.max_depth.is_some() && current_depth >= self.max_depth.unwrap())
            || (stop - start) <= self.min_samples_split
        {
            return vec![(oob_samples.to_vec(), mean)];
        }

        let mut best_gain = 0.;
        let mut best_split = 0;
        let mut best_split_val = 0.;
        let mut best_feature = 0;
        let mut left_sum_at_best_split = 0.;

        let mut feature_order = (0..self.X.ncols()).collect::<Vec<usize>>();
        feature_order.shuffle(rng);
        for (feature_idx, &feature) in feature_order.iter().enumerate() {
            // Note that we continue splitting until at least on non-constant feature
            // was evaluated.
            if (feature_idx as u16 >= self.mtry) && best_gain > 0. {
                break;
            }

            if constant_features[feature] {
                continue;
            }

            let (split, split_val, gain, left_sum) =
                self.find_best_split(start, stop, feature, sum);

            if gain < self.min_gain_to_split {
                constant_features[feature_idx] = true;
            } else if gain > best_gain {
                best_gain = gain;
                best_split = split;
                best_split_val = split_val;
                best_feature = feature;
                left_sum_at_best_split = left_sum;
            }
        }

        if best_gain <= self.min_gain_to_split {
            return vec![(oob_samples.to_vec(), mean)];
        }

        self.split_samples(
            start,
            best_split,
            stop,
            &constant_features,
            best_feature,
            best_split_val,
        );

        let (left_oob_samples, right_oob_samples) =
            split_oob_samples(oob_samples, self.X, best_feature, best_split_val);

        let mut left = self.split(
            start,
            best_split,
            left_oob_samples,
            constant_features.clone(),
            current_depth + 1,
            Some(left_sum_at_best_split),
            rng,
        );
        let mut right = self.split(
            best_split,
            stop,
            right_oob_samples,
            constant_features,
            current_depth + 1,
            Some(sum - left_sum_at_best_split),
            rng,
        );
        left.append(&mut right);
        left
    }

    // Calculate mean sum of y[samples[0][start..stop]].
    fn sum(&self, start: usize, stop: usize) -> f64 {
        let mut sum = 0.;
        for idx in self.samples[0][start..stop].iter() {
            sum += self.y[*idx];
        }
        sum
    }

    /// Find the best split in `self.X[start..stop, feature]`.
    pub fn find_best_split(
        &self,
        start: usize,
        stop: usize,
        feature: usize,
        sum: f64,
    ) -> (usize, f64, f64, f64) {
        let samples = &self.samples[feature];
        // X is constant in this segment.
        if self.X[[samples[stop - 1], feature]] - self.X[[samples[start], feature]] < 1e-12 {
            return (0, 0., 0., 0.);
        }

        let mut cumsum = 0.;
        let mut max_proxy_gain = 0.;
        let mut proxy_gain: f64;
        let mut split = start;
        let mut left_sum: f64 = 0.;

        for s in 1..(self.min_samples_leaf) {
            cumsum += self.y[samples[s - 1]];
        }

        for s in (start + self.min_samples_leaf)..(stop - self.min_samples_leaf + 1) {
            cumsum += self.y[samples[s - 1]];

            // Hackedy hack.
            if self.X[[samples[s], feature]] - self.X[[samples[s - 1], feature]] < 1e-12 {
                continue;
            }

            // Inspired by https://github.com/scikit-learn/scikit-learn/blob/cb4688ad15f052d7c55b1d3f09ee65bc3d5bb24b/sklearn/tree/_criterion.pyx#L900
            // The RSS after fitting a mean to (u, v] is L(u, v) = sum_{i=u+1}^v (y_i - mean)^2.
            // Here mean = 1 / (v - u) * sum_{i=u+1}^v y_i.
            // Then L(u, v) = \sum_{i=u+1}^v y_i^2 - 1 / (v - u) (sum_{i=u+1} y_i)^2.
            // The node impurity splitting at s is
            // L(start, s) + L(s, stop) = \sum_{i=start+1}^stop y_i^2 - 1 / (s - start) (sum_{i=start+1}^v y_i)^2 - 1 / (stop - s) (sum_{i=s+1}^stop y_i)^2.
            // The first term is independent of s, so does not need to be calculated to find the best split.
            // We find the maximum of the negative of the second term, which is the proxy gain.
            proxy_gain = cumsum * cumsum / (s - start) as f64
                + (sum - cumsum) * (sum - cumsum) / (stop - s) as f64;
            if proxy_gain > max_proxy_gain {
                max_proxy_gain = proxy_gain;
                split = s;
                left_sum = cumsum;
            }
        }

        // We are interested in the gain when splitting at s, the improvement in impurity
        // through splitting: G(s) = L(start, stop) - L(start, s) - L(s, stop).
        // The gain is always non-negative. If its maximum value is zero, then y is constant
        // on (start, stop). Then
        // G(s) = - 1 / (stop - start) * (\sum_{i=start+1}^stop y_i) ^ 2 + proxy_gain(s).
        // We also normalize by (stop - start).
        let max_gain =
            -(sum / (stop - start) as f64).powi(2) + max_proxy_gain / (stop - start) as f64;
        let split_val: f64;

        if split == start {
            (0, 0., 0., 0.)
        } else {
            split_val =
                self.X[[samples[split], feature]] / 2. + self.X[[samples[split - 1], feature]] / 2.;
            (split, split_val, max_gain, left_sum)
        }
    }

    /// Reorder `samples[feature][start..stop]` for each non-constant feature `feature`
    /// s.t. indices `samples[feature][start..split]`
    /// point to observations that belong to the left node (i.e. have
    /// `x[best_feature] <= best_split_val`) and indices `samples[feature][split..stop]`
    /// point to observations that belong to the right node,
    /// while preserving that `self.X[start..split, samples[features][start..split]` and
    /// `self.X[split..stop, samples[features][split..stop]]` are sorted.
    fn split_samples(
        &mut self,
        start: usize,
        split: usize,
        stop: usize,
        constant_features: &[bool],
        best_feature: usize,
        best_split_val: f64,
    ) {
        // Either store left or right samples in temporary vector, depending on which
        // is smaller. The others are done in-place.
        if stop - split <= split - start {
            let mut right_temp = Vec::<usize>::with_capacity(stop - split);

            let mut samples: &mut [usize];
            let mut current_left: usize;
            for (feature, &is_constant) in constant_features.iter().enumerate() {
                if feature == best_feature || is_constant {
                    continue;
                }
                samples = &mut self.samples[feature];
                // https://stackoverflow.com/a/10334085/10586763
                // Even digits in the example correspond to indices belonging to the right
                // node, odd digits to the left.

                // samples[start, .., current_left) contains (sorted by X) indices belonging
                // to the left node.
                current_left = start;

                for idx in start..stop {
                    if self.X[[samples[idx], best_feature]] > best_split_val {
                        //right_temp.push(samples[idx]);
                        right_temp.push(samples[idx]);
                    } else {
                        samples[current_left] = samples[idx];
                        current_left += 1;
                    }
                }
                self.samples[feature][split..stop].copy_from_slice(&right_temp);
                right_temp.clear()
            }
        } else {
            let mut left_temp: Vec<usize> = vec![0; split - start]; //Vec::<usize>::with_capacity(split - start);

            let mut samples: &mut [usize];
            let mut current_right: usize;

            for (feature, &is_constant) in constant_features.iter().enumerate() {
                if feature == best_feature || is_constant {
                    continue;
                }
                samples = &mut self.samples[feature];

                // samples[split, .., current_right) contains (sorted by X) indices belonging
                // to the right node.
                current_right = stop;
                let mut idx_left_temp = split - start;

                for idx in (start..stop).rev() {
                    if self.X[[samples[idx], best_feature]] > best_split_val {
                        current_right -= 1;
                        samples[current_right] = samples[idx];
                    } else {
                        idx_left_temp -= 1;
                        left_temp[idx_left_temp] = samples[idx];
                        //left_temp.push(samples[idx]);
                    }
                }
                self.samples[feature][start..split].copy_from_slice(&left_temp);
            }
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
    use assert_approx_eq::*;
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

        let (split, split_val, gain, _) =
            tree.find_best_split(start, stop, feature, y.slice(s![start..stop]).sum());

        assert_eq!((expected_split, expected_split_val), (split, split_val));

        assert_approx_eq!(expected_gain, gain);
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
        #[case] best_feature: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(0);
        let X = Array::random_using((100, 10), Uniform::new(0., 1.), &mut rng);
        let X_view = X.view();
        let y = Array::random_using(100, Uniform::new(0., 1.), &mut rng);
        let y_view = y.view();

        let all_false = vec![false; X.ncols()];
        let mut tree = DecisionTree::default(&X_view, &y_view);

        // Separate samples s.t. `tree.samples[feature_idx][start..stop]` contains the
        // same indices for each `feature_idx`.
        if start > 0 {
            let x_sorted = X
                .column(best_feature)
                .select(Axis(0), &tree.samples[best_feature]);
            let best_split_val = x_sorted[start] / 2. + x_sorted[start - 1] / 2.;
            tree.split_samples(0, start, 100, &all_false, best_feature, best_split_val);
        }

        if stop < 100 {
            let x_sorted = X
                .column(best_feature)
                .select(Axis(0), &tree.samples[best_feature]);
            let best_split_val = x_sorted[stop] / 2. + x_sorted[stop - 1] / 2.;
            tree.split_samples(start, stop, 100, &all_false, best_feature, best_split_val);
        }

        let x_sorted = X
            .column(best_feature)
            .select(Axis(0), &tree.samples[best_feature]);
        let best_split_val = x_sorted[split] / 2. + x_sorted[split - 1] / 2.;

        let samples_copy = tree.samples.clone();

        tree.split_samples(start, split, stop, &all_false, best_feature, best_split_val);

        for feature in 0..X.ncols() {
            assert!(is_sorted(
                &X.column(feature)
                    .select(Axis(0), &tree.samples[feature][start..split])
            ));
            assert!(is_sorted(
                &X.column(feature)
                    .select(Axis(0), &tree.samples[feature][split..stop])
            ));

            for idx in tree.samples[feature][start..split].iter() {
                assert!(X[[*idx, best_feature]] <= best_split_val);
            }

            for idx in tree.samples[feature][split..stop].iter() {
                assert!(X[[*idx, best_feature]] > best_split_val);
            }

            let mut before = samples_copy[feature][start..stop].to_vec();
            before.sort();
            let mut after = tree.samples[feature][start..stop].to_vec();
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
        let mut rng = StdRng::seed_from_u64(0);
        let result = tree.split(
            0,
            X.nrows(),
            &mut oob_samples,
            vec![false; 4],
            0,
            None,
            &mut rng,
        );

        let mut predictions = Array1::zeros(stop - start);
        for (idxs, val) in result.iter() {
            for idx in idxs {
                predictions[idx - start] = *val;
            }
        }

        let mse = (predictions - y.slice(s![start..stop]))
            .mapv(|x| x * x)
            .sum();
        assert!(mse <= 2., "Got mse of {}.", mse);
    }
}
