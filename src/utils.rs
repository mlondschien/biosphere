use rand::Rng;

/// Compute `indices` such that `data[indices]` is sorted.
#[allow(dead_code)]
pub fn argsort<T>(data: &[T]) -> Vec<usize>
where
    T: std::cmp::PartialOrd,
{
    let mut indices = (0..data.len()).collect::<Vec<usize>>();
    indices.sort_unstable_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
}

#[allow(dead_code)]
pub fn sample_weights(n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut counts = vec![0; n];

    for _ in 0..n {
        counts[rng.gen_range(0..n)] += 1
    }

    counts
}

#[allow(dead_code)]
pub fn sample_indices_from_weights(weights: &[usize], indices: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut samples = Vec::<Vec<usize>>::with_capacity(indices.len());

    for idx in 0..indices.len() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::is_sorted;
    use ndarray::{s, Array, Axis};
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

        let indices = argsort(&x.as_slice().unwrap());
        assert!(is_sorted(x.select(Axis(0), &indices).as_slice().unwrap()));
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
        let d = 5;

        let X = Array::random_using((n, d), Uniform::new(0., 1.), &mut rng);
        let mut indices = Vec::<Vec<usize>>::with_capacity(d);

        for idx in 0..d {
            indices.push(argsort(&X.slice(s![.., idx]).to_vec()));
        }

        let weights = sample_weights(n, &mut rng);

        let samples = sample_indices_from_weights(&weights, &indices);

        for idx in 0..d {
            assert!(is_sorted(
                X.slice(s![.., idx])
                    .select(Axis(0), &samples[idx])
                    .as_slice()
                    .unwrap()
            ));
        }
    }
}
