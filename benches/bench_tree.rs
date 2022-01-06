use biosphere::utils::{
    argsort, oob_samples_from_weights, sample_indices_from_weights, sample_weights,
};
use biosphere::{DecisionTree, DecisionTreeParameters};

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
pub fn benchmark_tree(c: &mut Criterion) {
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut group = c.benchmark_group("tree_split");

    for (n, d, max_depth, mtry) in &[
        (100, 10, 4, 10),
        (1000, 10, 4, 10),
        (10000, 10, 4, 10),
        (10000, 10, 16, 10),
        (10000, 100, 4, 10),
        (10000, 100, 4, 100),
        (100000, 10, 4, 10),
    ] {
        let (X, y) = data(*n, *d, &mut rng);

        let X_view = X.view();
        let y_view = y.view();

        let weights = sample_weights(*n, &mut rng);
        let indices: Vec<Vec<usize>> = (0..*d).map(|col| argsort(&X.column(col))).collect();
        let samples = sample_indices_from_weights(&weights, &indices);
        let mut oob_samples = oob_samples_from_weights(&weights);

        let decision_tree_parameters = DecisionTreeParameters::default()
            .with_max_depth(Some(*max_depth))
            .with_mtry(Some(*mtry));
        group.bench_function(
            format!(
                "tree_n={}, d={}, max_depth={}, mtry={}",
                n, d, max_depth, mtry
            )
            .as_str(),
            |b| {
                b.iter(|| {
                    let mut tree = DecisionTree::new(
                        &X_view,
                        &y_view,
                        samples.clone(),
                        decision_tree_parameters.clone(),
                    );
                    tree.split(0, *n, &mut oob_samples, vec![false; *d], 0, None, &mut rng)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(bench_tree, benchmark_tree);

criterion_main!(bench_tree);
