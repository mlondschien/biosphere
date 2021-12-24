use ndarray::{ArrayBase, Data, Ix1};
use rand::Rng;

/// Compute `indices` such that `data.select(indices)` is sorted.
///
/// Parameters
/// ----------
/// data: Array1<f64> or ArrayView<f64>
pub fn argsort(data: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<usize>>();
    indices.sort_unstable_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
}

pub fn sample_weights(n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut counts = vec![0; n];

    for _ in 0..n {
        counts[rng.gen_range(0..n)] += 1
    }

    counts
}

pub fn sample_indices_from_weights(
    weights: &[usize],
    indices: &[Vec<usize>],
    features: &[usize],
) -> Vec<Vec<usize>> {
    let mut samples = Vec::<Vec<usize>>::with_capacity(features.len());

    for &idx in features {
        let mut sample = Vec::<usize>::with_capacity(indices[idx].len());
        for jdx in 0..indices[idx].len() {
            for _ in 0..weights[indices[idx][jdx]] {
                sample.push(indices[idx][jdx]);
            }
        }
        samples.push(sample);
    }
    samples
}

pub fn oob_samples_from_weights(weights: &[usize]) -> Vec<usize> {
    let mut oob_samples = Vec::<usize>::with_capacity(weights.len());

    for (idx, &weight) in weights.iter().enumerate() {
        if weight == 0 {
            oob_samples.push(idx);
        }
    }
    oob_samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::is_sorted;
    use ndarray::{Array, Axis};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_argsort() {
        let seed = 7;
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 100;
        let x = Array::random_using(n, Uniform::new(0., 1.), &mut rng);

        let indices = argsort(&x);
        assert!(is_sorted(&x.select(Axis(0), &indices)));
    }

    #[test]
    fn test_sample_weights() {
        let seed = 7;
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 100;

        let weights = sample_weights(n, &mut rng);

        assert!(weights.iter().sum::<usize>() == n);
    }

    #[test]
    fn test_sample_indices_from_weights() {
        let seed = 7;
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 100;
        let d = 8;

        let X = Array::random_using((n, d), Uniform::new(0., 1.), &mut rng);

        let indices: Vec<Vec<usize>> = (0..d).map(|idx| argsort(&X.column(idx))).collect();
        let weights = sample_weights(n, &mut rng);

        let features = &[0, 2, 3, 4, 6];
        let samples = sample_indices_from_weights(&weights, &indices, features);

        for (feature_idx, &feature) in features.iter().enumerate() {
            assert!(is_sorted(
                &X.column(feature).select(Axis(0), &samples[feature_idx])
            ));
        }
    }
}
