use criterion::{criterion_group, criterion_main, Criterion};
use oobforest::RandomForest;

#[cfg(test)]
use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub fn data(seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = 10000;
    let X = Array::random_using((n, 100), Uniform::new(0., 1.), &mut rng);
    let y = Array::random_using(n, Uniform::new(0., 1.), &mut rng);
    let y = y + X.column(0) + X.column(1).map(|x| x - x * x);

    (X, y)
}

#[allow(non_snake_case)]
pub fn criterion_benchmark(c: &mut Criterion) {
    let seed = 0;
    let (X, y) = data(seed);

    let X_view = X.view();
    let y_view = y.view();
    let forest = RandomForest::new(&X_view, &y_view, None, Some(8), None, None, None, None);
    c.bench_function("forest", |b| b.iter(|| forest.predict()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
