use crate::tree::DecisionTreeParameters;
use rand::seq::SliceRandom;
use rand::Rng;
use std::debug_assert;

static MIN_GAIN_TO_SPLIT: f64 = 1e-12;
static FEATURE_THRESHOLD: f64 = 1e-14;

#[derive(Default)]
pub struct DecisionTreeNode {
    pub left_child: Option<Box<DecisionTreeNode>>,
    pub right_child: Option<Box<DecisionTreeNode>>,
    pub feature_index: Option<usize>,
    pub feature_value: Option<f64>,
    pub label: Option<f64>,
}

impl DecisionTreeNode {
    fn leaf_node(&mut self, label: f64) {
        self.label = Some(label);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split(
        &mut self,
        // For each feature fidx, xy_sorted[fidx][i] = (X[[row, fidx]], y[row], row),
        // where row = argsort(X.column(fidx))[i]. That is, X[[row, fidx]] is sorted.
        xy_sorted: Vec<&mut [(f64, f64, usize)]>,
        n_samples: usize,
        mut constant_features: Vec<bool>,
        // Used in split_samples. Passed here to avoid reallocating.
        all_false: &mut [bool],
        sum: f64,
        rng: &mut impl Rng,
        current_depth: usize,
        parameters: &DecisionTreeParameters,
    ) {
        if parameters.max_depth.is_some_and(|d| current_depth >= d)
            || n_samples <= parameters.min_samples_split
        {
            return self.leaf_node(sum / n_samples as f64);
        }

        let mut best_gain = 0.;
        let mut best_split = 0;
        let mut best_split_val = 0.;
        let mut best_feature = 0;
        let mut left_sum_at_best_split = 0.;

        let n_features = xy_sorted.len();
        let mut feature_order = (0..n_features).collect::<Vec<usize>>();
        feature_order.shuffle(rng);

        for (feature_idx, &feature) in feature_order.iter().enumerate() {
            // We continue splitting until at least one non-constant feature was evaluated.
            if feature_idx >= parameters.max_features.from_n_features(n_features) && best_gain > 0.
            {
                break;
            }

            if constant_features[feature] {
                continue;
            }

            // X[, feature] is constant on this segment.
            if xy_sorted[feature].last().unwrap().0 - xy_sorted[feature].first().unwrap().0
                < FEATURE_THRESHOLD
            {
                constant_features[feature] = true;
                continue;
            }

            let (split, split_val, gain, left_sum) = self.find_best_split(xy_sorted[feature], sum);

            if gain > best_gain {
                best_gain = gain;
                best_split = split;
                best_split_val = split_val;
                best_feature = feature;
                left_sum_at_best_split = left_sum;
            }
        }

        if best_gain < MIN_GAIN_TO_SPLIT {
            return self.leaf_node(sum / n_samples as f64);
        }

        let (left_xy_sorted, right_xy_sorted) = self.split_samples(
            xy_sorted,
            best_split,
            &constant_features,
            best_feature,
            all_false,
        );

        let mut left = DecisionTreeNode::default();
        left.split(
            left_xy_sorted,
            best_split,
            constant_features.clone(),
            all_false,
            left_sum_at_best_split,
            rng,
            current_depth + 1,
            parameters,
        );
        self.left_child = Some(Box::new(left));

        let mut right = DecisionTreeNode::default();
        right.split(
            right_xy_sorted,
            n_samples - best_split,
            constant_features,
            all_false,
            sum - left_sum_at_best_split,
            rng,
            current_depth + 1,
            parameters,
        );
        self.right_child = Some(Box::new(right));

        self.feature_index = Some(best_feature);
        self.feature_value = Some(best_split_val);
    }

    /// Find the best split point. `xy_sorted[i] = (x_val, y_val, row)` are the (x, y, row)
    /// triples for this feature in sorted order of x.
    fn find_best_split(&self, xy_sorted: &[(f64, f64, usize)], sum: f64) -> (usize, f64, f64, f64) {
        let n = xy_sorted.len();
        let mut cumsum = 0.;
        let mut max_proxy_gain = 0.;
        let mut proxy_gain: f64;
        let mut split = 0;
        let mut left_sum: f64 = 0.;

        for s in 1..n {
            let (x_prev, y_prev, _) = xy_sorted[s - 1];
            let x_cur = xy_sorted[s].0;

            debug_assert!(x_cur >= x_prev);

            cumsum += y_prev;

            if x_cur - x_prev < 1e-12 {
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
            proxy_gain =
                cumsum * cumsum / s as f64 + (sum - cumsum) * (sum - cumsum) / (n - s) as f64;

            if proxy_gain > max_proxy_gain {
                max_proxy_gain = proxy_gain;
                split = s;
                left_sum = cumsum;
            }
        }

        debug_assert!((cumsum + xy_sorted[n - 1].1 - sum).abs() < 1e-12);

        // We are interested in the gain when splitting at s, the improvement in impurity
        // through splitting: G(s) = L(start, stop) - L(start, s) - L(s, stop).
        // The gain is always non-negative. If its maximum value is zero, then y is constant
        // on (start, stop). Then
        // G(s) = - 1 / (stop - start) * (\sum_{i=start+1}^stop y_i) ^ 2 + proxy_gain(s).
        // We also normalize by (stop - start).
        let max_gain = -(sum / n as f64).powi(2) + max_proxy_gain / n as f64;

        if split == 0 {
            (0, 0., 0., 0.)
        } else {
            let split_val = xy_sorted[split].0 / 2. + xy_sorted[split - 1].0 / 2.;
            (split, split_val, max_gain, left_sum)
        }
    }

    /// Split xy_sorted into left/right halves, maintaining sorted order within each half.
    ///
    /// For each feature, xy_sorted[feature] contains (x, y, row) triples sorted by x.
    /// split_samples partitions each feature's slice so that rows with
    /// X[row, best_feature] <= best_split_val go left and the rest go right,
    /// preserving sorted order within each half.
    fn split_samples<'a>(
        &self,
        xy_sorted: Vec<&'a mut [(f64, f64, usize)]>,
        split: usize,
        constant_features: &[bool],
        best_feature: usize,
        all_false: &mut [bool],
    ) -> (
        Vec<&'a mut [(f64, f64, usize)]>,
        Vec<&'a mut [(f64, f64, usize)]>,
    ) {
        // Mark right-going rows using the row index stored in xy_sorted.
        for (_, _, idx) in xy_sorted[best_feature][split..].iter() {
            all_false[*idx] = true;
        }

        let n = xy_sorted[best_feature].len();
        let mut left_out = Vec::with_capacity(xy_sorted.len());
        let mut right_out = Vec::with_capacity(xy_sorted.len());

        let mut first_left: &mut [(f64, f64, usize)] = &mut [];
        let mut copy_of_first_right: Vec<(f64, f64, usize)> = Vec::with_capacity(n - split);
        let mut initialized = false;
        let mut index_of_first: usize = 0;

        let mut right_scratch: &mut [(f64, f64, usize)] = &mut [];

        let mut current_left: usize;
        let mut current_right: usize;

        for (feature, xy_) in xy_sorted.into_iter().enumerate() {
            if feature == best_feature {
                let (l, r) = xy_.split_at_mut(split);
                left_out.push(l);
                right_out.push(r);
                continue;
            }

            if constant_features[feature] {
                left_out.push(&mut []);
                right_out.push(&mut []);
                continue;
            }

            if !initialized {
                let (l, r) = xy_.split_at_mut(split);
                right_scratch = r;
                copy_of_first_right.extend_from_slice(right_scratch);
                first_left = l;
                index_of_first = feature;
                initialized = true;
                continue;
            }

            // https://stackoverflow.com/a/10334085/10586763
            // Even digits in the example correspond to indices belonging to the right
            // node, odd digits to the left.

            // xy_[..current_left) contains (sorted by X) triples belonging to the left node.
            current_left = 0;
            current_right = 0;

            for i in 0..n {
                if all_false[xy_[i].2] {
                    right_scratch[current_right] = xy_[i];
                    current_right += 1;
                } else {
                    xy_[current_left] = xy_[i];
                    current_left += 1;
                }
            }

            let (l, r) = xy_.split_at_mut(split);
            left_out.push(l);
            right_out.push(right_scratch);
            right_scratch = r;
        }

        if initialized {
            current_left = 0;
            current_right = 0;

            for i in 0..split {
                if all_false[first_left[i].2] {
                    right_scratch[current_right] = first_left[i];
                    current_right += 1;
                } else {
                    first_left[current_left] = first_left[i];
                    current_left += 1;
                }
            }

            for i in 0..(n - split) {
                if all_false[copy_of_first_right[i].2] {
                    right_scratch[current_right] = copy_of_first_right[i];
                    current_right += 1;
                } else {
                    first_left[current_left] = copy_of_first_right[i];
                    current_left += 1;
                }
            }
            left_out.insert(index_of_first, first_left);
            right_out.insert(index_of_first, right_scratch);
        }

        // Reset all_false to be all false.
        for (_, _, idx) in right_out[best_feature].iter() {
            all_false[*idx] = false;
        }

        (left_out, right_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::is_sorted;
    use assert_approx_eq::*;
    use ndarray::{arr1, arr2, s, Array, Array1, Axis};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rstest::*;

    #[rstest]
    #[case(&[0., 0., 0., 1., 1., 1.], 0, 6, 0, 3, 2.5, 0.25)]
    #[case(&[0., 0., 0., 1., 1., 1.], 1, 5, 0, 2, 2.5, 0.25)]
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

        let node = DecisionTreeNode::default();
        let xy_sorted: Vec<(f64, f64, usize)> = (start..stop)
            .map(|i| (X_view[[i, feature]], y_view[i], i))
            .collect();

        let (split, split_val, gain, _) =
            node.find_best_split(&xy_sorted, y.slice(s![start..stop]).sum());

        assert_eq!((expected_split, expected_split_val), (split, split_val));

        assert_approx_eq!(expected_gain, gain);
    }

    #[test]
    fn test_find_trivial_best_split() {
        let mut rng = StdRng::seed_from_u64(0);
        let X = Array::random_using((100, 1), Uniform::new(0., 1.).unwrap(), &mut rng);
        let y = Array1::<f64>::zeros(100);

        let node = DecisionTreeNode::default();
        let mut indices = (0..100).collect::<Vec<usize>>();
        indices.sort_unstable_by(|&a, &b| X[[a, 0]].partial_cmp(&X[[b, 0]]).unwrap());

        let xy_sorted: Vec<(f64, f64, usize)> =
            indices.iter().map(|&i| (X[[i, 0]], y[i], i)).collect();

        let (split, split_val, gain, sum) = node.find_best_split(&xy_sorted, 0.);
        assert_eq!((split, split_val, gain, sum), (0, 0., 0., 0.));
    }

    #[rstest]
    #[case(50, 1, 0.5, 5)]
    #[case(100, 2, 0.1, 5)]
    #[case(100, 0, 0.1, 1)]
    #[case(100, 5, 1., 10)]
    #[case(100, 5, 0.2, 10)]
    #[case(500, 5, 0.2, 10)]
    fn test_split_samples(
        #[case] n_samples: usize,
        #[case] best_feature: usize,
        #[case] best_split_val: f64,
        #[case] d: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(0);
        let X = Array::random_using((100, d), Uniform::new(0., 1.).unwrap(), &mut rng);
        let y = Array::random_using(100, Uniform::new(0., 1.).unwrap(), &mut rng);

        let mut single_samples: Vec<usize> =
            (0..n_samples).map(|_| rng.random_range(0..100)).collect();
        single_samples.sort();

        // Build xy_sorted directly by sorting (x, y, row) triples per feature.
        let mut xy_sorted_vecs: Vec<Vec<(f64, f64, usize)>> = (0..d)
            .map(|f| {
                let mut v: Vec<(f64, f64, usize)> = single_samples
                    .iter()
                    .map(|&i| (X[[i, f]], y[i], i))
                    .collect();
                v.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                v
            })
            .collect();

        let split = X
            .column(best_feature)
            .select(Axis(0), &single_samples)
            .iter()
            .filter(|&&x| x <= best_split_val)
            .count();

        let xy_sorted_refs: Vec<&mut [(f64, f64, usize)]> = xy_sorted_vecs
            .iter_mut()
            .map(|v| v.as_mut_slice())
            .collect();

        let mut all_false_but_first = vec![false; X.ncols()];
        all_false_but_first[0] = true;

        let node = DecisionTreeNode::default();
        let mut all_false = vec![false; X.nrows()];

        let (left, right) = node.split_samples(
            xy_sorted_refs,
            split,
            &all_false_but_first,
            best_feature,
            &mut all_false,
        );

        assert!(left.len() == d);
        assert!(right.len() == d);

        for (feature, (l, r)) in left.into_iter().zip(right).enumerate().skip(1) {
            let l_rows: Vec<usize> = l.iter().map(|t| t.2).collect();
            let r_rows: Vec<usize> = r.iter().map(|t| t.2).collect();

            assert!(is_sorted(&X.column(feature).select(Axis(0), &l_rows)));
            assert!(is_sorted(&X.column(feature).select(Axis(0), &r_rows)));

            for row in l_rows.iter() {
                assert!(X[[*row, best_feature]] <= best_split_val);
            }

            for row in r_rows.iter() {
                assert!(X[[*row, best_feature]] > best_split_val);
            }

            let mut all_rows = [l_rows, r_rows].concat();
            all_rows.sort();

            assert_eq!(all_rows, single_samples);
        }
    }
}
