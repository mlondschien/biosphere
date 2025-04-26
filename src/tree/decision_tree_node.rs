use crate::tree::DecisionTreeParameters;
use ndarray::{ArrayView1, ArrayView2};
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
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        samples: Vec<&mut [usize]>,
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

        let mut feature_order = (0..X.ncols()).collect::<Vec<usize>>();
        feature_order.shuffle(rng);

        for (feature_idx, &feature) in feature_order.iter().enumerate() {
            // Note that we continue splitting until at least on non-constant feature
            // was evaluated.
            if feature_idx >= parameters.max_features.from_n_features(X.ncols()) && best_gain > 0. {
                break;
            }

            if constant_features[feature] {
                continue;
            }

            // X[, feature] is constant on this segment.
            if X[[*samples[feature].last().unwrap(), feature]]
                - X[[*samples[feature].first().unwrap(), feature]]
                < FEATURE_THRESHOLD
            {
                constant_features[feature] = true;
                continue;
            }

            let (split, split_val, gain, left_sum) =
                self.find_best_split(X, y, feature, samples[feature], sum);

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

        let (left_samples, right_samples) = self.split_samples(
            samples,
            best_split,
            &constant_features,
            best_feature,
            all_false,
        );

        let mut left = DecisionTreeNode::default();
        left.split(
            X,
            y,
            left_samples,
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
            X,
            y,
            right_samples,
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

    /// Find the best split in `self.X[samples, feature]`.
    fn find_best_split(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        feature: usize,
        samples: &[usize],
        sum: f64,
    ) -> (usize, f64, f64, f64) {
        let n = samples.len();
        let mut cumsum = 0.;
        let mut max_proxy_gain = 0.;
        let mut proxy_gain: f64;
        let mut split = 0;
        let mut left_sum: f64 = 0.;

        for s in 1..samples.len() {
            debug_assert!(X[[samples[s], feature]] >= X[[samples[s - 1], feature]]);

            cumsum += y[samples[s - 1]];

            // Hackedy hack.
            if X[[samples[s], feature]] - X[[samples[s - 1], feature]] < 1e-12 {
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

        debug_assert!((cumsum + y[*samples.last().unwrap()] - sum).abs() < 1e-12);

        // We are interested in the gain when splitting at s, the improvement in impurity
        // through splitting: G(s) = L(start, stop) - L(start, s) - L(s, stop).
        // The gain is always non-negative. If its maximum value is zero, then y is constant
        // on (start, stop). Then
        // G(s) = - 1 / (stop - start) * (\sum_{i=start+1}^stop y_i) ^ 2 + proxy_gain(s).
        // We also normalize by (stop - start).
        let max_gain = -(sum / n as f64).powi(2) + max_proxy_gain / n as f64;

        let split_val: f64;

        if split == 0 {
            (0, 0., 0., 0.)
        } else {
            split_val = X[[samples[split], feature]] / 2. + X[[samples[split - 1], feature]] / 2.;
            (split, split_val, max_gain, left_sum)
        }
    }

    /// Split samples into two, corresponding to observations left / right of the split point.
    ///
    /// `samples` is a vector of slices. For each feature s.t. constant_features[feature]
    /// is false, samples[feature] are indices s.t. X[samples[feature], feature] is
    /// sorted.
    ///
    /// split_samples takes each of these slices and divides them into left (for indices
    /// s s.t. X[s, best_feature] <= best_split_val) / right (others), making sure that
    /// X[left, feature] and X[right, feature] are still ordered.
    fn split_samples<'a>(
        &self,
        samples: Vec<&'a mut [usize]>,
        split: usize,
        constant_features: &[bool],
        best_feature: usize,
        // best_split_val: f64,
        all_false: &mut [bool],
    ) -> (Vec<&'a mut [usize]>, Vec<&'a mut [usize]>) {
        // We replace lookups & comparisons X[[idx, best_feature]] > best_split_val
        // with a lookup all_false[idx]. This is faster. Since best_feature was split
        // at best_split_val, the comparison holds true exactly for samples after split.
        for s in samples[best_feature][split..].iter() {
            all_false[*s] = true;
        }

        let n = samples[best_feature].len();
        let mut new_samples_left = Vec::<&mut [usize]>::with_capacity(samples.len());
        let mut new_samples_right = Vec::<&mut [usize]>::with_capacity(samples.len());

        let mut first_left: &mut [usize] = &mut [];
        let mut copy_of_first_right: Vec<usize> = Vec::with_capacity(n - split);
        let mut initialized = false;
        let mut index_of_first: usize = 0;

        let mut new_right: &mut [usize] = &mut [];

        let mut current_left: usize;
        let mut current_right: usize;

        for (feature, sample_) in samples.into_iter().enumerate() {
            if feature == best_feature {
                let (left, right) = sample_.split_at_mut(split);
                new_samples_left.push(left);
                new_samples_right.push(right);
                continue;
            }

            if constant_features[feature] {
                new_samples_left.push(&mut []);
                new_samples_right.push(&mut []);
                continue;
            }

            if !initialized {
                let result = sample_.split_at_mut(split);
                new_right = result.1;
                copy_of_first_right.extend_from_slice(new_right);
                first_left = result.0;
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
                    current_right += 1;
                } else {
                    sample_[current_left] = sample_[idx];
                    current_left += 1;
                }
            }

            let result = sample_.split_at_mut(split);
            new_samples_left.push(result.0);
            new_samples_right.push(new_right);
            new_right = result.1;
        }

        if initialized {
            current_left = 0;
            current_right = 0;

            for idx in 0..split {
                // if X[[first_left[idx], best_feature]] > best_split_val {
                if all_false[first_left[idx]] {
                    new_right[current_right] = first_left[idx];
                    current_right += 1;
                } else {
                    first_left[current_left] = first_left[idx];
                    current_left += 1;
                }
            }

            for idx in 0..(n - split) {
                // if X[[copy_of_first_right[idx], best_feature]] > best_split_val {
                if all_false[copy_of_first_right[idx]] {
                    new_right[current_right] = copy_of_first_right[idx];
                    current_right += 1;
                } else {
                    first_left[current_left] = copy_of_first_right[idx];
                    current_left += 1;
                }
            }
            new_samples_left.insert(index_of_first, first_left);
            new_samples_right.insert(index_of_first, new_right);
        }

        // Reset all_false to be all false.
        for s in new_samples_right[best_feature].iter() {
            all_false[*s] = false;
        }

        (new_samples_left, new_samples_right)
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
        let (split, split_val, gain, _) = node.find_best_split(
            &X_view,
            &y_view,
            feature,
            &samples,
            y.slice(s![start..stop]).sum(),
        );

        assert_eq!((expected_split, expected_split_val), (split, split_val));

        assert_approx_eq!(expected_gain, gain);
    }

    #[test]
    fn test_find_trivial_best_split() {
        let mut rng = StdRng::seed_from_u64(0);
        let X = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
        let y = Array1::<f64>::zeros(100);

        let node = DecisionTreeNode::default();
        let mut samples = (0..100).collect::<Vec<usize>>();
        samples.sort_unstable_by(|a, b| X[[*a, 0]].partial_cmp(&X[[*b, 0]]).unwrap());

        let (split, split_val, gain, sum) =
            node.find_best_split(&X.view(), &y.view(), 0, &samples, 0.);
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
        let X = Array::random_using((100, d), Uniform::new(0., 1.), &mut rng);

        let mut single_samples: Vec<usize> =
            (0..n_samples).map(|_| rng.gen_range(0..100)).collect();
        single_samples.sort();
        let mut samples = sorted_samples(&X, &single_samples);

        let split = X
            .column(best_feature)
            .select(Axis(0), &single_samples)
            .iter()
            .filter(|&&x| x <= best_split_val)
            .count();
        let samples_references = samples.iter_mut().map(|x| x.as_mut_slice()).collect();

        let mut all_false_but_first = vec![false; X.ncols()];
        all_false_but_first[0] = true;

        let node = DecisionTreeNode::default();
        let mut all_false = vec![false; X.nrows()];

        let (left, right) = node.split_samples(
            samples_references,
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
