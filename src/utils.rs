use ndarray::{ArrayBase, Data, Ix1, Ix2};
use rand::Rng;

/// Compute `indices` such that `data.select(indices)` is sorted.
///
/// Parameters
/// ----------
/// data: Array1<f64> or ArrayView1<f64>
pub fn argsort(data: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<usize>>();
    indices.sort_unstable_by(|a, b| data[*a].partial_cmp(&data[*b]).unwrap());
    indices
}

pub fn sample_weights(n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut counts = vec![0; n];

    for _ in 0..n {
        counts[rng.gen_range(0..n)] += 1
    }

    counts
}

pub fn sample_indices_from_weights(weights: &[usize], indices: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut samples = Vec::<Vec<usize>>::with_capacity(indices.len());

    for feature_indices in indices {
        let mut sample = Vec::<usize>::with_capacity(feature_indices.len());
        for &index in feature_indices {
            for _ in 0..weights[index] {
                sample.push(index);
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

pub fn sorted_samples(
    X: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    samples: &[usize],
) -> Vec<Vec<usize>> {
    let mut samples_out: Vec<Vec<usize>> = Vec::with_capacity(X.ncols());

    for idx in 0..X.ncols() {
        let mut samples_ = samples.to_vec();
        samples_.sort_by(|a, b| X[[*a, idx]].partial_cmp(&X[[*b, idx]]).unwrap());
        samples_out.push(samples_);
    }
    samples_out
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

        let samples = sample_indices_from_weights(&weights, &indices);

        for feature in 0..X.ncols() {
            assert!(is_sorted(
                &X.column(feature).select(Axis(0), &samples[feature])
            ));
        }
    }
}
