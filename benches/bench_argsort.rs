use biosphere::utils::argsort;
#[cfg(test)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array;
use ndarray_rand::rand_distr::{Bernoulli, Uniform};
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub fn benchmark_argsort(c: &mut Criterion) {
    let seed = 0;
    let mut group = c.benchmark_group("argsort");
    let sizes: &[usize] = &[1000, 10000, 100000, 1000000];
    for &size in sizes.iter() {
        let mut rng = StdRng::seed_from_u64(seed);
        let x = Array::random_using(size, Uniform::new(0., 1.), &mut rng);
        group.bench_with_input(BenchmarkId::new("argsort_continuous", size), &x, |b, x| {
            b.iter(|| argsort(&x))
        });
        let y = Array::random_using(size, Bernoulli::new(0.3).unwrap(), &mut rng)
            .mapv(|x| x as i64 as f64);
        group.bench_with_input(BenchmarkId::new("argsort_one_hot", size), &y, |b, x| {
            b.iter(|| argsort(&x))
        });
    }
}

criterion_group!(bench_argsort, benchmark_argsort);
criterion_main!(bench_argsort);
