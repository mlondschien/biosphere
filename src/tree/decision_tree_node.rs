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
        samples: Vec<&mut [usize]>,
        // For each feature f, xy_sorted[f][i] = (X[[samples[f][i], f]], y[samples[f][i]]),
        // i.e. the (x, y) pairs gathered in sorted order of feature f.
        xy_sorted: Vec<&mut [(f64, f64)]>,
        n_samples: usize,
        mut constant_features: Vec<bool>,
        // Used in split_samples. Passed here to avoid reallocating.
        all_false: &mut [bool],
        sum: f64,
        rng: &mut impl Rng,
        current_depth: usize,
        parameters: &DecisionTreeParameters,
    ) {
        if let Some(depth) = parameters.max_depth {
            if current_depth >= depth {
                return self.leaf_node(sum / n_samples as f64);
            }
        }

        if n_samples <= parameters.min_samples_split {
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
            // Note that we continue splitting until at least one non-constant feature
            // was evaluated.
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

        if best_gain <= MIN_GAIN_TO_SPLIT {
            return self.leaf_node(sum / n_samples as f64);
        }

        let (left_samples, right_samples, left_xy_sorted, right_xy_sorted) = self.split_samples(
            samples,
            xy_sorted,
            best_split,
            &constant_features,
            best_feature,
            all_false,
        );

        let mut left = DecisionTreeNode::default();
        left.split(
            left_samples,
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
            right_samples,
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

    /// Find the best split point. `xy_sorted[i] = (x_val, y_val)` are the (x, y) pairs
    /// for this feature in sorted order of x. Both values are read sequentially with no
    /// random memory access.
    fn find_best_split(&self, xy_sorted: &[(f64, f64)], sum: f64) -> (usize, f64, f64, f64) {
        let n = xy_sorted.len();
        let mut cumsum = 0.;
        let mut max_proxy_gain = 0.;
        let mut proxy_gain: f64;
        let mut split = 0;
        let mut left_sum: f64 = 0.;

        for s in 1..n {
            let (x_prev, y_prev) = xy_sorted[s - 1];
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

    /// Split samples into two, corresponding to observations left / right of the split point.
    ///
    /// `samples` is a vector of slices. For each feature s.t. constant_features[feature]
    /// is false, samples[feature] are indices s.t. X[samples[feature], feature] is sorted.
    /// `xy_sorted[feature]` contains (x, y) pairs in that same sorted order.
    ///
    /// split_samples takes each of these slices and divides them into left (for indices
    /// s s.t. X[s, best_feature] <= best_split_val) / right (others), keeping both
    /// samples and xy_sorted in sync and maintaining sorted order within each half.
    fn split_samples<'a>(
        &self,
        samples: Vec<&'a mut [usize]>,
        xy_sorted: Vec<&'a mut [(f64, f64)]>,
        split: usize,
        constant_features: &[bool],
        best_feature: usize,
        // best_split_val: f64,
        all_false: &mut [bool],
    ) -> (
        Vec<&'a mut [usize]>,
        Vec<&'a mut [usize]>,
        Vec<&'a mut [(f64, f64)]>,
        Vec<&'a mut [(f64, f64)]>,
    ) {
        // We replace lookups & comparisons X[[idx, best_feature]] > best_split_val
        // with a lookup all_false[idx]. This is faster. Since best_feature was split
        // at best_split_val, the comparison holds true exactly for samples after split.
        for s in samples[best_feature][split..].iter() {
            all_false[*s] = true;
        }

        let n = samples[best_feature].len();
        let mut new_samples_left = Vec::<&mut [usize]>::with_capacity(samples.len());
        let mut new_samples_right = Vec::<&mut [usize]>::with_capacity(samples.len());
        let mut new_xy_sorted_left = Vec::<&mut [(f64, f64)]>::with_capacity(xy_sorted.len());
        let mut new_xy_sorted_right = Vec::<&mut [(f64, f64)]>::with_capacity(xy_sorted.len());

        let mut first_left: &mut [usize] = &mut [];
        let mut first_left_xy: &mut [(f64, f64)] = &mut [];
        let mut copy_of_first_right: Vec<usize> = Vec::with_capacity(n - split);
        let mut copy_of_first_right_xy: Vec<(f64, f64)> = Vec::with_capacity(n - split);
        let mut initialized = false;
        let mut index_of_first: usize = 0;

        let mut new_right: &mut [usize] = &mut [];
        let mut new_right_xy: &mut [(f64, f64)] = &mut [];

        let mut current_left: usize;
        let mut current_right: usize;

        for (feature, (sample_, xy_sorted_)) in
            samples.into_iter().zip(xy_sorted.into_iter()).enumerate()
        {
            if feature == best_feature {
                let (left, right) = sample_.split_at_mut(split);
                let (left_xy, right_xy) = xy_sorted_.split_at_mut(split);
                new_samples_left.push(left);
                new_samples_right.push(right);
                new_xy_sorted_left.push(left_xy);
                new_xy_sorted_right.push(right_xy);
                continue;
            }

            if constant_features[feature] {
                new_samples_left.push(&mut []);
                new_samples_right.push(&mut []);
                new_xy_sorted_left.push(&mut []);
                new_xy_sorted_right.push(&mut []);
                continue;
            }

            if !initialized {
                let result = sample_.split_at_mut(split);
                let result_xy = xy_sorted_.split_at_mut(split);
                new_right = result.1;
                new_right_xy = result_xy.1;
                copy_of_first_right.extend_from_slice(new_right);
                copy_of_first_right_xy.extend_from_slice(new_right_xy);
                first_left = result.0;
                first_left_xy = result_xy.0;
                index_of_first = feature;
                initialized = true;
                continue;
            }

            // https://stackoverflow.com/a/10334085/10586763
            // Even digits in the example correspond to indices belonging to the right
            // node, odd digits to the left.

            // samples[..current_left) contains (sorted by X) indices belonging
            // to the left node.
            current_left = 0;
            current_right = 0;

            for idx in 0..n {
                // if X[[sample_[idx], best_feature]] > best_split_val {
                if all_false[sample_[idx]] {
                    new_right[current_right] = sample_[idx];
                    new_right_xy[current_right] = xy_sorted_[idx];
                    current_right += 1;
                } else {
                    sample_[current_left] = sample_[idx];
                    xy_sorted_[current_left] = xy_sorted_[idx];
                    current_left += 1;
                }
            }

            let result = sample_.split_at_mut(split);
            let result_xy = xy_sorted_.split_at_mut(split);
            new_samples_left.push(result.0);
            new_samples_right.push(new_right);
            new_xy_sorted_left.push(result_xy.0);
            new_xy_sorted_right.push(new_right_xy);
            new_right = result.1;
            new_right_xy = result_xy.1;
        }

        if initialized {
            current_left = 0;
            current_right = 0;

            for idx in 0..split {
                // if X[[first_left[idx], best_feature]] > best_split_val {
                if all_false[first_left[idx]] {
                    new_right[current_right] = first_left[idx];
                    new_right_xy[current_right] = first_left_xy[idx];
                    current_right += 1;
                } else {
                    first_left[current_left] = first_left[idx];
                    first_left_xy[current_left] = first_left_xy[idx];
                    current_left += 1;
                }
            }

            for idx in 0..(n - split) {
                // if X[[copy_of_first_right[idx], best_feature]] > best_split_val {
                if all_false[copy_of_first_right[idx]] {
                    new_right[current_right] = copy_of_first_right[idx];
                    new_right_xy[current_right] = copy_of_first_right_xy[idx];
                    current_right += 1;
                } else {
                    first_left[current_left] = copy_of_first_right[idx];
                    first_left_xy[current_left] = copy_of_first_right_xy[idx];
                    current_left += 1;
                }
            }
            new_samples_left.insert(index_of_first, first_left);
            new_samples_right.insert(index_of_first, new_right);
            new_xy_sorted_left.insert(index_of_first, first_left_xy);
            new_xy_sorted_right.insert(index_of_first, new_right_xy);
        }

        // Reset all_false to be all false.
        for s in new_samples_right[best_feature].iter() {
            all_false[*s] = false;
        }

        (
            new_samples_left,
            new_samples_right,
            new_xy_sorted_left,
            new_xy_sorted_right,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::is_sorted;
    use crate::utils::sorted_samples;
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
        let samples = (start..stop).collect::<Vec<usize>>();
        let xy_sorted: Vec<(f64, f64)> = samples
            .iter()
            .map(|&i| (X_view[[i, feature]], y_view[i]))
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
        let mut samples = (0..100).collect::<Vec<usize>>();
        samples.sort_unstable_by(|a, b| X[[*a, 0]].partial_cmp(&X[[*b, 0]]).unwrap());

        let xy_sorted: Vec<(f64, f64)> = samples.iter().map(|&i| (X[[i, 0]], y[i])).collect();

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
        let mut samples = sorted_samples(&X, &single_samples);
        let mut xy_sorted_vecs: Vec<Vec<(f64, f64)>> = samples
            .iter()
            .enumerate()
            .map(|(f, s)| s.iter().map(|&i| (X[[i, f]], y[i])).collect())
            .collect();

        let split = X
            .column(best_feature)
            .select(Axis(0), &single_samples)
            .iter()
            .filter(|&&x| x <= best_split_val)
            .count();
        let samples_references: Vec<&mut [usize]> =
            samples.iter_mut().map(|x| x.as_mut_slice()).collect();
        let xy_sorted_refs: Vec<&mut [(f64, f64)]> = xy_sorted_vecs
            .iter_mut()
            .map(|v| v.as_mut_slice())
            .collect();

        let mut all_false_but_first = vec![false; X.ncols()];
        all_false_but_first[0] = true;

        let node = DecisionTreeNode::default();
        let mut all_false = vec![false; X.nrows()];

        let (left, right, _left_xy, _right_xy) = node.split_samples(
            samples_references,
            xy_sorted_refs,
            split,
            &all_false_but_first,
            best_feature,
            &mut all_false,
        );

        assert!(left.len() == d);
        assert!(right.len() == d);

        for (feature, (l, r)) in left.into_iter().zip(right).enumerate().skip(1) {
            assert!(is_sorted(&X.column(feature).select(Axis(0), l)));
            assert!(is_sorted(&X.column(feature).select(Axis(0), r)));

            for idx in l.iter() {
                assert!(X[[*idx, best_feature]] <= best_split_val);
            }

            for idx in r.iter() {
                assert!(X[[*idx, best_feature]] > best_split_val);
            }

            let mut all_samples = [l, r].concat();
            all_samples.sort();

            assert_eq!(all_samples, single_samples);
        }
    }
}
