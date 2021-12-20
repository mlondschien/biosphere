use crate::argsort::argsort;
use ndarray::{s, Array1, Array2, Axis};

#[allow(dead_code)]
struct DecisionTree {
    X: Array2<f64>,
    y: Array1<f64>,
    max_depth: usize,
    features: Vec<usize>,
    // order: Vec<Array1<f64>>,
    // indices: Vec<usize>,
    // indices_oob: Vec<usize>,
}

impl DecisionTree {
    #[allow(dead_code)]
    fn find_best_split(&self, feature: usize, in_bag_indices: &[usize]) -> (usize, f64, f64) {
        let n = in_bag_indices.len();

        let x_ = self
            .X
            .slice(s![.., feature])
            .select(Axis(0), in_bag_indices);
        let indices = argsort(x_.as_slice().unwrap());
        let x_ = x_.select(Axis(0), &indices);
        let mut y_cumsum = self
            .y
            .select(Axis(0), in_bag_indices)
            .select(Axis(0), &indices);
        y_cumsum.accumulate_axis_inplace(Axis(0), |&prev, cur| *cur += prev);
        let mut max_gain = f64::MIN;
        let mut gain: f64;
        let mut split = 0;
        for s in 1..(n - 1) {
            gain = (s as f64 * y_cumsum[n - 1] - n as f64 * y_cumsum[s - 1]).powi(2)
                / (n * s * (n - s)) as f64;
            if gain > max_gain {
                max_gain = gain;
                split = s;
            }
        }
        let split_val = (x_[split] + x_[split - 1]) / 2.;
        (split, split_val, max_gain)
    }

    #[allow(unused)]
    fn split(
        &self,
        in_bag_indices: Vec<usize>,
        oob_indices: Vec<usize>,
        depth: usize,
    ) -> Vec<(Vec<usize>, f64)> {
        if depth >= self.max_depth || in_bag_indices.len() <= 2 {
            return self.calculate_mean(oob_indices);
        }

        let mut best_gain = f64::MIN;
        // let mut best_split = 0;
        let mut best_split_val = 0.;
        let mut best_feature = 0;

        for feature in self.features.iter() {
            let (split, split_val, gain) = self.find_best_split(*feature, &in_bag_indices);
            if gain > best_gain {
                best_gain = gain;
                // best_split = split;
                best_split_val = split_val;
                best_feature = *feature;
            }
        }

        if best_gain <= 0. {
            return self.calculate_mean(oob_indices);
        }

        let left_in_bag_indices = in_bag_indices
            .iter()
            .filter(|&&idx| self.X[[idx, best_feature]] < best_split_val)
            .cloned()
            .collect();
        let left_oob_indices = oob_indices
            .iter()
            .filter(|&&idx| self.X[[idx, best_feature]] < best_split_val)
            .cloned()
            .collect();
        let right_in_bag_indices = in_bag_indices
            .iter()
            .filter(|&&idx| self.X[[idx, best_feature]] >= best_split_val)
            .cloned()
            .collect();
        let right_oob_indices = oob_indices
            .iter()
            .filter(|&&idx| self.X[[idx, best_feature]] >= best_split_val)
            .cloned()
            .collect();
        let mut left = self.split(left_in_bag_indices, left_oob_indices, depth + 1);
        let mut right = self.split(right_in_bag_indices, right_oob_indices, depth + 1);
        left.append(&mut right);
        left
    }

    fn calculate_mean(&self, oob_indices: Vec<usize>) -> Vec<(Vec<usize>, f64)> {
        let mut mean_value = 0.;
        let n = oob_indices.len() as f64;
        for idx in &oob_indices {
            mean_value += self.y[*idx];
        }
        mean_value /= n;
        return vec![(oob_indices, mean_value)];
    }
}

#[cfg(test)]
mod tests {
    use super::DecisionTree;
    use ndarray::{arr1, arr2};
    use rstest::*;

    #[rstest]
    #[case(2, 1.5, 0)]
    #[case(3, 2.5, 1)]
    fn test_find_best_split(
        #[case] expected_split: usize,
        #[case] expected_split_val: f64,
        #[case] feature: usize,
    ) {
        let X = arr2(&[[0., 1., 2., 3., 4., 5.], [3., 3., 2., 1., 1., 4.]])
            .t()
            .to_owned();
        let y = arr1(&[0., 0., 2., 1., 2., 2.]);
        let tree = DecisionTree {
            X,
            y,
            max_depth: 1,
            features: vec![0, 1],
        };
        let (split, split_val, _) = tree.find_best_split(feature, &[0, 1, 2, 3, 4, 5]);

        assert_eq!(expected_split, split);

        assert_eq!(expected_split_val, split_val);
    }

    #[test]
    fn test_split() {
        let X = arr2(&[
            [0., 0.],
            [0., 2.],
            [0., 4.],
            [0., 5.],
            [1., 1.],
            [2., 2.],
            [2., 3.],
            [2., 4.],
            [3., 1.],
            [3., 5.],
            [4., 0.],
            [4., 2.],
            [4., 3.],
            [4., 4.],
        ]);
        let y = arr1(&[1., 1., 2., 2., 1., 1., 2., 2., 0., -1., 0., 0., -1., -1.]);

        let in_bag_indices = vec![0, 1, 2, 5, 6, 7, 8, 9, 11, 12];
        let oob_indices = vec![3, 4, 10, 13];

        let tree = DecisionTree {
            X,
            y,
            max_depth: 3,
            features: vec![0, 1],
        };

        let result = tree.split(in_bag_indices, oob_indices, 0);

        let mut predictions = vec![-7.; 14];
        for (idxs, prediction) in result.iter() {
            for idx in idxs {
                predictions[*idx] = *prediction;
            }
        }

        assert_eq!(
            predictions,
            vec![-7., -7., -7., 2.0, 1.0, -7., -7., -7., -7., -7., 0., -7., -7., -1.,]
        );
    }
}
