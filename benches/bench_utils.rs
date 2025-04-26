use biosphere::utils::{
    argsort, oob_samples_from_weights, sample_indices_from_weights, sample_weights,
};
#[cfg(test)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array;
use ndarray_rand::rand_distr::{Bernoulli, Uniform};
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub fn benchmark_utils(c: &mut Criterion) {
    let seed = 0;
    let mut group = c.benchmark_group("utils");
    let sizes: &[usize] = &[100000];
    for &size in sizes.iter() {
        let mut rng = StdRng::seed_from_u64(seed);
        let x = Array::random_using(size, Uniform::new(0., 1.), &mut rng);
        group.bench_with_input(BenchmarkId::new("argsort_continuous", size), &x, |b, x| {
            b.iter(|| argsort(x))
        });
        let y = Array::random_using(size, Bernoulli::new(0.3).unwrap(), &mut rng)
            .mapv(|x| x as i64 as f64);
        group.bench_with_input(BenchmarkId::new("argsort_one_hot", size), &y, |b, x| {
            b.iter(|| argsort(x))
        });
        group.bench_function(format!("sample_weight, size={}", size), |b| {
            b.iter(|| sample_weights(size, &mut rng))
        });
        let weights = sample_weights(size, &mut rng);
        let indices = vec![argsort(&x)];
        group.bench_function(format!("sample_indices_from_weights, size={}", size), |b| {
            b.iter(|| sample_indices_from_weights(&weights, &indices))
        });
        group.bench_function(format!("oob_samples_from_weights, size={}", size), |b| {
            b.iter(|| oob_samples_from_weights(&weights))
        });
    }
}

criterion_group!(bench_utils, benchmark_utils);
criterion_main!(bench_utils);
