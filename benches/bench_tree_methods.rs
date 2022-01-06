use biosphere::DecisionTree;

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
pub fn benchmark_tree_methods(c: &mut Criterion) {
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut group = c.benchmark_group("tree_methods");
    let d = 10;

    for &n in &[1000, 10000, 100000] {
        let (X, y) = data(n, 10, &mut rng);

        let X_view = X.view();
        let y_view = y.view();

        let tree = DecisionTree::default(&X_view, &y_view);
        let sum_0 = X_view.column(0).sum();
        let sum_1 = X_view.column(1).sum();

        group.bench_function(
            format!("find_best_split, continuous, n={}", n).as_str(),
            |b| b.iter(|| tree.find_best_split(0, n, 0, sum_0)),
        );

        group.bench_function(format!("find_best_split, one-hot, n={}", n).as_str(), |b| {
            b.iter(|| tree.find_best_split(0, n, 1, sum_1))
        });

        group.bench_function(format!("sum, n={}", n).as_str(), |b| {
            b.iter(|| tree.sum(0, n))
        });

        let split = X.column(0).iter().filter(|&x| *x <= 0.6).count();
        group.bench_function(format!("split_samples continuous, n={}", n).as_str(), |b| {
            b.iter(|| {
                let mut tree_clone = tree.clone();
                tree_clone.split_samples(0, split, n, &vec![false; d], 0, 0.6);
            })
        });
    }
    group.finish();
}

criterion_group!(bench_tree_methods, benchmark_tree_methods);

criterion_main!(bench_tree_methods);
