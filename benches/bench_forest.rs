#[cfg(test)]
#[allow(non_snake_case)]
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use oobforest::RandomForest;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

pub fn data(n: usize, d: usize, rng: &mut impl Rng) -> (Array2<f64>, Array1<f64>) {
    let X = Array::random_using((n, d), Uniform::new(0., 1.), rng);
    let y = Array::random_using(n, Uniform::new(0., 1.), rng);
    let y = y + X.column(0) + X.column(1).map(|x| x - x * x);

    (X, y)
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let seed = 0;
    let n = 10000;
    let d = 100;
    let mut rng = StdRng::seed_from_u64(seed);

    let (X, y) = data(n, d, &mut rng);

    let X_view = X.view();
    let y_view = y.view();
    let forest = RandomForest::new(&X_view, &y_view, None, Some(8), None, None, None, None);
    c.bench_function("forest", |b| b.iter(|| forest.predict()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
