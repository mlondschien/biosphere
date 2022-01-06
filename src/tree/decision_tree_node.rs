use crate::tree::DecisionTreeParameters;
use ndarray::{ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::Rng;

static MIN_GAIN_TO_SPLIT: f64 = 1e-12;

pub struct DecisionTreeNode {
    pub left_child: Option<Box<DecisionTreeNode>>,
    pub right_child: Option<Box<DecisionTreeNode>>,
    pub feature_index: Option<usize>,
    pub feature_value: Option<f64>,
    pub label: Option<f64>,
}

impl DecisionTreeNode {
    pub fn new() -> Self {
        DecisionTreeNode {
            left_child: None,
            right_child: None,
            feature_index: None,
            feature_value: None,
            label: None,
        }
    }

    fn leaf_node(&mut self, label: f64) {
        self.label = Some(label);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split(
        &mut self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        samples: Vec<&mut [usize]>,
        mut constant_features: Vec<bool>,
        sum: f64,
        rng: &mut impl Rng,
        current_depth: usize,
        parameters: &DecisionTreeParameters,
    ) {
        if let Some(depth) = parameters.max_depth {
            if current_depth >= depth {
                return self.leaf_node(sum / X.nrows() as f64);
            }
        }

        if samples[0].len() <= parameters.min_samples_split {
            return self.leaf_node(sum / X.nrows() as f64);
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
            if let Some(mtry) = parameters.mtry {
                if feature_idx >= mtry && best_gain > 0. {
                    break;
                }
            }

            if constant_features[feature] {
                continue;
            }

            let (split, split_val, gain, left_sum) =
                self.find_best_split(X, y, feature, samples[feature], sum);

            if gain <= MIN_GAIN_TO_SPLIT {
                constant_features[feature_idx] = true;
            } else if gain > best_gain {
                best_gain = gain;
                best_split = split;
                best_split_val = split_val;
                best_feature = feature;
                left_sum_at_best_split = left_sum;
            }
        }

        if best_gain <= 0. {
            return self.leaf_node(sum / X.nrows() as f64);
        }

        let (left_samples, right_samples) = self.split_samples(
            X,
            samples,
            best_split,
            &constant_features,
            best_feature,
            best_split_val,
        );

        let mut left = DecisionTreeNode::new();
        left.split(
            X,
            y,
            left_samples,
            constant_features.clone(),
            left_sum_at_best_split,
            rng,
            current_depth + 1,
            parameters,
        );
        self.left_child = Some(Box::new(left));

        let mut right = DecisionTreeNode::new();
        right.split(
            X,
            y,
            right_samples,
            constant_features,
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
    pub fn find_best_split(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        feature: usize,
        samples: &[usize],
        sum: f64,
    ) -> (usize, f64, f64, f64) {
        // X is constant in this segment.
        if X[[*samples.last().unwrap(), feature]] - X[[*samples.first().unwrap(), feature]] < 1e-12
        {
            return (0, 0., 0., 0.);
        }

        let n = samples.len();
        let mut cumsum = 0.;
        let mut max_proxy_gain = 0.;
        let mut proxy_gain: f64;
        let mut split = 0;
        let mut left_sum: f64 = 0.;

        for s in 1..samples.len() {
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

    /// Reorder `samples[feature][start..stop]` for each non-constant feature `feature`
    /// s.t. indices `samples[feature][start..split]`
    /// point to observations that belong to the left node (i.e. have
    /// `x[best_feature] <= best_split_val`) and indices `samples[feature][split..stop]`
    /// point to observations that belong to the right node,
    /// while preserving that `self.X[start..split, samples[features][start..split]` and
    /// `self.X[split..stop, samples[features][split..stop]]` are sorted.
    pub fn split_samples<'a>(
        &self,
        X: &ArrayView2<f64>,
        samples: Vec<&'a mut [usize]>,
        split: usize,
        constant_features: &[bool],
        best_feature: usize,
        best_split_val: f64,
    ) -> (Vec<&'a mut [usize]>, Vec<&'a mut [usize]>) {
        let n = samples[best_feature].len();
        let mut new_samples_left = Vec::<&mut [usize]>::with_capacity(samples.len());
        let mut new_samples_right = Vec::<&mut [usize]>::with_capacity(samples.len());

        // Either store left or right samples in temporary vector, depending on which
        // is smaller. The others are done in-place.
        // if split > n / 2 {
        let mut right_temp = Vec::<usize>::with_capacity(n - split);
        let mut current_left: usize;

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

            // https://stackoverflow.com/a/10334085/10586763
            // Even digits in the example correspond to indices belonging to the right
            // node, odd digits to the left.

            // samples[..current_left) contains (sorted by X) indices belonging
            // to the left node.
            current_left = 0;

            for idx in 0..n {
                if X[[sample_[idx], best_feature]] > best_split_val {
                    right_temp.push(sample_[idx]);
                } else {
                    sample_[current_left] = sample_[idx];
                    current_left += 1;
                }
            }

            sample_[split..].copy_from_slice(&right_temp);
            let (left, right) = sample_.split_at_mut(split);
            new_samples_left.push(left);
            new_samples_right.push(right);
            right_temp.clear()
        }

        (new_samples_left, new_samples_right)
    }
}
