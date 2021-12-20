/// Check if input is sorted. Used for testing.
///
/// From https://stackoverflow.com/questions/51272571/how-do-i-check-if-a-slice-is-sorted.
#[allow(dead_code)]
fn is_sorted<T>(data: &[T]) -> bool
where
    T: std::cmp::PartialOrd,
{
    data.windows(2).all(|w| w[0] <= w[1])
}

/// Compute `indices` such that `data[indices]` is sorted.
#[allow(dead_code)]
fn argsort<T>(data: &[T]) -> Vec<usize>
where
    T: std::cmp::PartialOrd,
{
    let mut indices = (0..data.len()).collect::<Vec<usize>>();
    indices.sort_unstable_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
}

#[cfg(test)]
mod tests {
    use super::{argsort, is_sorted};
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
}
