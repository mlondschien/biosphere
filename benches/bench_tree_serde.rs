use biosphere::{DecisionTree, DecisionTreeParameters, MaxFeatures};

#[cfg(test)]
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::hint::black_box;

#[allow(non_snake_case)]
pub fn data(n: usize, d: usize, rng: &mut impl Rng) -> (Array2<f64>, Array1<f64>) {
    let X = Array2::from_shape_fn((n, d), |_| rng.random::<f64>());

    let y = Array1::from_shape_fn(n, |_| rng.random::<f64>());
    let y = y + X.column(0) + X.column(1).map(|x| x - x * x);

    (X, y)
}

#[allow(non_snake_case)]
pub fn benchmark_tree_fit_vs_deserialize(c: &mut Criterion) {
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);

    let n = 10_000usize;
    let d = 10usize;
    let max_depth = 8usize;

    let (X, y) = data(n, d, &mut rng);
    let X_view = X.view();
    let y_view = y.view();

    let parameters = DecisionTreeParameters::default()
        .with_max_depth(Some(max_depth))
        .with_max_features(MaxFeatures::Value(d))
        .with_random_state(seed);

    let mut fitted_tree = DecisionTree::new(parameters.clone());
    fitted_tree.fit(&X_view, &y_view);
    let fitted_tree_bytes = postcard::to_stdvec(&fitted_tree).unwrap();

    let mut group = c.benchmark_group("tree_fit_vs_deserialize");
    group.bench_function(format!("fit_n={n}, d={d}, max_depth={max_depth}"), |b| {
        b.iter(|| {
            let mut tree = DecisionTree::new(parameters.clone());
            tree.fit(&X_view, &y_view);
            black_box(tree);
        })
    });

    group.bench_function(
        format!("deserialize_n={n}, d={d}, max_depth={max_depth}"),
        |b| {
            b.iter(|| {
                let tree: DecisionTree =
                    postcard::from_bytes(black_box(fitted_tree_bytes.as_slice())).unwrap();
                black_box(tree);
            })
        },
    );

    group.finish();
}

criterion_group!(
    name = tree_serde;
    config = Criterion::default().sample_size(10);
    targets = benchmark_tree_fit_vs_deserialize
);

criterion_main!(tree_serde);
