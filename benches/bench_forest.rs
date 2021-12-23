#[allow(non_snake_case)]
use criterion::{criterion_group, criterion_main, Criterion};
use oobforest::RandomForest;

use csv::ReaderBuilder;
use ndarray::{s, Array2};
use ndarray_csv::Array2Reader;
use std::fs::File;

pub fn iris() -> Array2<f64> {
    let file = File::open("testdata/iris.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let data: Array2<f64> = reader.deserialize_array2((150, 5)).unwrap();
    data
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let data = iris();
    let X = data.slice(s![.., 0..4]);
    let y = data.slice(s![.., 4]);

    let forest = RandomForest::new(&X, &y, None, Some(8), None, None, None, None);

    c.bench_function("forest", |b| b.iter(|| forest.predict()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
