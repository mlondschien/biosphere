use biosphere::RandomForest;
#[cfg(test)]
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

#[allow(non_snake_case)]
pub fn data(n: usize, d: usize, rng: &mut impl Rng) -> (Array2<f64>, Array1<f64>) {
    let X = Array::random_using((n, d), Uniform::new(0., 1.), rng);
    let y = Array::random_using(n, Uniform::new(0., 1.), rng);
    let y = y + X.column(0) + X.column(1).map(|x| x - x * x);

    (X, y)
}

#[allow(non_snake_case)]
pub fn criterion_benchmark(c: &mut Criterion) {
    let seed = 0;
    let n = 50000;
    let d = 10;
    let mut rng = StdRng::seed_from_u64(seed);

    let (X, y) = data(n, d, &mut rng);

    let X_view = X.view();
    let y_view = y.view();
    let forest = RandomForest::new(
        &X_view,
        &y_view,
        None,
        Some(16),
        None,
        None,
        None,
        None,
        None,
    );
    c.bench_function("forest", |b| b.iter(|| forest.predict()));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark
);
criterion_main!(benches);
