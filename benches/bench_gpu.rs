/// Benchmarks comparing CPU and GPU inference for the same trained forest.
///
/// Run with:
///   cargo bench --features gpu --bench bench_gpu
///
/// The benchmark group "predict" varies batch size across three backends:
///   - `RandomForest`  — original tree-of-structs CPU inference
///   - `FlatForest`    — flat BFS array CPU inference
///   - `GpuForest`     — GPU inference via wgpu compute shaders
///
/// Throughput is reported in samples/s so the numbers are directly comparable
/// across batch sizes.
use biosphere::gpu::GpuForest;
use biosphere::{FlatForest, RandomForest, RandomForestParameters};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;

const N_TRAIN: usize = 1_000;
const N_FEATURES: usize = 20;
/// Batch sizes to sweep. Larger sizes favour the GPU; smaller sizes favour CPU.
const SAMPLE_SIZES: &[usize] = &[100, 1_000, 10_000, 100_000];

#[allow(non_snake_case)]
fn make_X(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::random_using((n, N_FEATURES), Uniform::new(0.0, 1.0).unwrap(), &mut rng)
}

#[allow(non_snake_case)]
fn benchmark_predict(c: &mut Criterion) {
    // --- Train once, outside the timed loop ---
    let X_train = make_X(N_TRAIN, 0);
    let y_train = X_train.column(0).to_owned();

    let params = RandomForestParameters::default()
        .with_n_estimators(100)
        .with_max_depth(Some(8))
        .with_seed(42);

    let mut forest = RandomForest::new(params);
    forest.fit(&X_train.view(), &y_train.view());

    let flat = FlatForest::from_forest(&forest, N_FEATURES);

    let max_n = *SAMPLE_SIZES.iter().max().unwrap();
    let gpu = GpuForest::from_flat_forest(&flat, max_n).unwrap();

    let mut group = c.benchmark_group("predict");

    for &n in SAMPLE_SIZES {
        let X_infer = make_X(n, 1);
        // Pre-convert to f32 once; the conversion itself is not under measurement.
        let X_f32 = X_infer.mapv(|v| v as f32);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("RandomForest", n), &n, |b, _| {
            b.iter(|| forest.predict(&X_infer.view()))
        });

        group.bench_with_input(BenchmarkId::new("FlatForest", n), &n, |b, _| {
            b.iter(|| flat.predict(&X_f32.view()))
        });

        group.bench_with_input(BenchmarkId::new("GpuForest", n), &n, |b, _| {
            b.iter(|| gpu.predict(&X_f32.view()))
        });
    }

    group.finish();
}

criterion_group!(
    name = gpu_bench;
    config = Criterion::default().sample_size(10);
    targets = benchmark_predict
);

criterion_main!(gpu_bench);
