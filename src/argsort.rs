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
pub fn sample_sorted_with_replacement(n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut counts: Vec<usize> = vec![0; n];
    let mut sample: Vec<usize> = Vec::with_capacity(n);

    for _ in 0..n {
        counts[rng.gen_range(0..n)] += 1
    }

    for (idx, val) in counts.iter().enumerate() {
        sample.append(&mut vec![idx; *val]);
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::{argsort, sample_sorted_with_replacement};
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

        let indices = argsort(&x.as_slice().unwrap());
        assert!(is_sorted(x.select(Axis(0), &indices).as_slice().unwrap()));
    }

    #[test]
    fn test_sample_sorted_with_replacement() {
        let seed = 7;
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 100;

        let sample = sample_sorted_with_replacement(n, &mut rng);

        assert!(is_sorted(&sample));
    }
}
