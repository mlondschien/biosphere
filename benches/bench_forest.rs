use biosphere::{RandomForest, RandomForestParameters};

#[cfg(test)]
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{s, Array, Array1, Array2};
use ndarray_rand::rand_distr::{Bernoulli, Uniform};
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

#[allow(non_snake_case)]
pub fn data(n: usize, d: usize, rng: &mut impl Rng) -> (Array2<f64>, Array1<f64>) {
    let mut X = Array2::<f64>::zeros((n, d));

    for i in 0..d {
        if i % 2 == 0 {
            X.slice_mut(s![.., i])
                .assign(&Array::random_using((n,), Uniform::new(0., 1.), rng));
        } else {
            X.slice_mut(s![.., i]).assign(
                &Array::random_using((n,), Bernoulli::new(0.3).unwrap(), rng)
                    .mapv(|x| x as i64 as f64),
            );
        }
    }
    let X = Array::random_using((n, d), Uniform::new(0., 1.), rng);
    let y = Array::random_using(n, Uniform::new(0., 1.), rng);
    let y = y + X.column(0) + X.column(1).map(|x| x - x * x);

    (X, y)
}

#[allow(non_snake_case)]
pub fn benchmark_forest(c: &mut Criterion) {
    let seed = 0;
    let n = 100000;
    let d = 10;
    let mut rng = StdRng::seed_from_u64(seed);

    let (X, y) = data(n, d, &mut rng);

    let X_view = X.view();
    let y_view = y.view();
    let random_forest_parameters = RandomForestParameters::default().with_max_depth(Some(4));

    let mut forest = RandomForest::new(random_forest_parameters.clone().with_n_jobs(Some(1)));
    c.bench_function("forest", |b| {
        b.iter(|| forest.fit_predict_oob(&X_view, &y_view))
    });

    let mut forest = RandomForest::new(random_forest_parameters.with_n_jobs(Some(4)));
    c.bench_function("forest_4_jobs", |b| {
        b.iter(|| forest.fit_predict_oob(&X_view, &y_view))
    });
}

criterion_group!(
    name = forest;
    config = Criterion::default().sample_size(10);
    targets = benchmark_forest
);

criterion_main!(forest);
