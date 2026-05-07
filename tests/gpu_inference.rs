/// GPU inference correctness: GpuForest::predict must match FlatForest::predict
/// within f32 precision.
///
/// Requires a real GPU. This test will panic if no GPU adapter is available.
#[cfg(feature = "gpu")]
mod gpu_tests {
    #![allow(non_snake_case)]
    use biosphere::gpu::GpuForest;
    use biosphere::{FlatForest, RandomForest, RandomForestParameters};
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_data(
        n_samples: usize,
        n_features: usize,
        seed: u64,
    ) -> (Array2<f64>, ndarray::Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let X = Array2::random_using(
            (n_samples, n_features),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut rng,
        );
        // Use first feature as target to avoid cumsum precision issues during training.
        let y = X.column(0).to_owned();
        (X, y)
    }

    #[test]
    fn gpu_matches_flat_cpu() {
        let n_features = 10;
        let (train_x, train_y) = make_data(150, n_features, 42);

        let params = RandomForestParameters::default()
            .with_n_estimators(20)
            .with_seed(1);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let (test_x, _) = make_data(100, n_features, 99);
        let flat = FlatForest::from_forest(&forest, n_features);
        let test_x_f32 = test_x.mapv(|v| v as f32);
        let cpu_preds = flat.predict(&test_x_f32.view());

        let gpu_forest = GpuForest::from_flat_forest(&flat, 100).unwrap();
        let gpu_preds = gpu_forest.predict(&test_x_f32.view()).unwrap();

        // Both FlatForest and GpuForest use f32 nodes and f32 comparisons.
        // The only difference is f64 vs f32 accumulation; tolerance ~1e-5.
        for (i, (cpu, gpu)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
            let tol = 1e-5;
            assert!(
                ((*cpu as f32) - gpu).abs() < tol,
                "sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
    }

    #[test]
    fn gpu_single_tree_single_sample() {
        let n_features = 4;
        let (train_x, train_y) = make_data(100, n_features, 7);

        let params = RandomForestParameters::default()
            .with_n_estimators(1)
            .with_seed(0);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let (test_x, _) = make_data(1, n_features, 55);
        let flat = FlatForest::from_forest(&forest, n_features);
        let test_x_f32 = test_x.mapv(|v| v as f32);
        let cpu_preds = flat.predict(&test_x_f32.view());

        let gpu_forest = GpuForest::from_flat_forest(&flat, 1).unwrap();
        let gpu_preds = gpu_forest.predict(&test_x_f32.view()).unwrap();

        assert!(
            ((cpu_preds[0] as f32) - gpu_preds[0]).abs() < 1e-5,
            "cpu={}, gpu={}",
            cpu_preds[0],
            gpu_preds[0]
        );
    }

    // --- Test #8/#9: n_samples at workgroup boundary values ---
    /// Tests 63, 64, 65, 127, 128, 129 samples — exercises the bounds check in the
    /// traverse shader and the exact-workgroup-multiple fast paths.
    #[test]
    fn gpu_workgroup_boundaries() {
        let n_features = 5;
        let (train_x, train_y) = make_data(150, n_features, 42);

        let params = RandomForestParameters::default()
            .with_n_estimators(10)
            .with_seed(1);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        // Allocate enough capacity for the largest batch we'll test.
        let gpu_forest = GpuForest::from_flat_forest(&flat, 200).unwrap();

        for &n_samples in &[1usize, 63, 64, 65, 127, 128, 129] {
            let (test_x, _) = make_data(n_samples, n_features, n_samples as u64 + 100);
            let test_x_f32 = test_x.mapv(|v| v as f32);

            let cpu_preds = flat.predict(&test_x_f32.view());
            let gpu_preds = gpu_forest.predict(&test_x_f32.view()).unwrap();

            assert_eq!(
                gpu_preds.len(),
                n_samples,
                "n_samples={n_samples}: output length mismatch"
            );
            for (i, (cpu, gpu)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
                assert!(
                    ((*cpu as f32) - gpu).abs() < 1e-5,
                    "n_samples={n_samples} sample {i}: cpu={cpu}, gpu={gpu}"
                );
            }
        }
    }

    // --- Test #10: multiple predict() calls on the same GpuForest ---
    /// Guards against accidental state mutation between calls.
    #[test]
    fn gpu_multiple_predict_calls() {
        let n_features = 6;
        let (train_x, train_y) = make_data(150, n_features, 77);

        let params = RandomForestParameters::default()
            .with_n_estimators(15)
            .with_seed(3);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let gpu_forest = GpuForest::from_flat_forest(&flat, 100).unwrap();

        let (test_x1, _) = make_data(80, n_features, 10);
        let (test_x2, _) = make_data(60, n_features, 20);
        let test_x1_f32 = test_x1.mapv(|v| v as f32);
        let test_x2_f32 = test_x2.mapv(|v| v as f32);

        let cpu1 = flat.predict(&test_x1_f32.view());
        let cpu2 = flat.predict(&test_x2_f32.view());

        let gpu1 = gpu_forest.predict(&test_x1_f32.view()).unwrap();
        let gpu2 = gpu_forest.predict(&test_x2_f32.view()).unwrap();

        for (i, (cpu, gpu)) in cpu1.iter().zip(gpu1.iter()).enumerate() {
            assert!(
                ((*cpu as f32) - gpu).abs() < 1e-5,
                "call 1 sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
        for (i, (cpu, gpu)) in cpu2.iter().zip(gpu2.iter()).enumerate() {
            assert!(
                ((*cpu as f32) - gpu).abs() < 1e-5,
                "call 2 sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
    }

    // --- Test #11: large n_trees (500) ---
    /// Stresses the y-dispatch dimension and the per-tree-preds buffer size.
    #[test]
    fn gpu_large_n_trees() {
        let n_features = 5;
        let (train_x, train_y) = make_data(150, n_features, 99);

        let params = RandomForestParameters::default()
            .with_n_estimators(500)
            .with_seed(4);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let (test_x, _) = make_data(50, n_features, 200);
        let test_x_f32 = test_x.mapv(|v| v as f32);

        let cpu_preds = flat.predict(&test_x_f32.view());
        let gpu_forest = GpuForest::from_flat_forest(&flat, 50).unwrap();
        let gpu_preds = gpu_forest.predict(&test_x_f32.view()).unwrap();

        for (i, (cpu, gpu)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
            assert!(
                ((*cpu as f32) - gpu).abs() < 1e-5,
                "sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
    }

    // --- is_uma() ---
    /// is_uma() must be callable and return a bool without panicking.
    /// Its value depends on the adapter type (true on Apple Silicon, false on discrete GPUs).
    #[test]
    fn gpu_is_uma_returns_bool() {
        let n_features = 4;
        let (train_x, train_y) = make_data(50, n_features, 1);
        let params = RandomForestParameters::default()
            .with_n_estimators(5)
            .with_seed(1);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let gpu_forest = GpuForest::from_flat_forest(&flat, 10).unwrap();

        // Just ensure it's callable; the value is adapter-dependent.
        let _uma: bool = gpu_forest.is_uma();
    }

    // --- with_collect_timeout ---
    /// Configuring a generous timeout must not break normal inference.
    #[test]
    fn gpu_with_collect_timeout_works() {
        let n_features = 4;
        let (train_x, train_y) = make_data(100, n_features, 2);
        let params = RandomForestParameters::default()
            .with_n_estimators(10)
            .with_seed(2);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let gpu_forest = GpuForest::from_flat_forest(&flat, 50)
            .unwrap()
            .with_collect_timeout(std::time::Duration::from_secs(60));

        let (test_x, _) = make_data(30, n_features, 99);
        let test_x_f32 = test_x.mapv(|v| v as f32);

        let cpu_preds = flat.predict(&test_x_f32.view());
        let gpu_preds = gpu_forest.predict(&test_x_f32.view()).unwrap();

        for (i, (cpu, gpu)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
            assert!(
                ((*cpu as f32) - gpu).abs() < 1e-5,
                "sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
    }

    // --- Busy flag: double predict_submit panics ---
    /// Calling predict_submit twice without an intervening collect must panic.
    #[test]
    #[should_panic(expected = "outstanding PredictHandle")]
    fn gpu_double_predict_submit_panics() {
        let n_features = 4;
        let (train_x, train_y) = make_data(50, n_features, 3);
        let params = RandomForestParameters::default()
            .with_n_estimators(5)
            .with_seed(3);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let gpu_forest = GpuForest::from_flat_forest(&flat, 20).unwrap();

        let (test_x, _) = make_data(10, n_features, 50);
        let test_x_f32 = test_x.mapv(|v| v as f32);

        let _handle = gpu_forest.predict_submit(&test_x_f32.view()).unwrap();
        // Second submit without collecting _handle — should panic.
        let _ = gpu_forest.predict_submit(&test_x_f32.view());
    }

    // --- Fix 5: double-submit guard test ---
    /// Verifies the busy-flag panic message contains "outstanding PredictHandle".
    /// Uses a minimal dataset (10 samples, 3 features, 5 trees) to keep it fast.
    #[test]
    #[should_panic(expected = "outstanding PredictHandle")]
    fn gpu_double_submit_panics() {
        let n_features = 3;
        let (train_x, train_y) = make_data(50, n_features, 99);
        let params = RandomForestParameters::default()
            .with_n_estimators(5)
            .with_seed(99);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let gpu_forest = GpuForest::from_flat_forest(&flat, 10).unwrap();

        let (test_x, _) = make_data(10, n_features, 200);
        let test_x_f32 = test_x.mapv(|v| v as f32);

        // First submit — OK.
        let _handle_a = gpu_forest.predict_submit(&test_x_f32.view()).unwrap();
        // Second submit without collecting _handle_a — must panic.
        let _ = gpu_forest.predict_submit(&test_x_f32.view());
    }

    // --- Test #10: GpuForest::fork ---
    /// A forked handle shares pipelines and node data with the original but has
    /// independent buffers. Both must produce identical predictions.
    #[test]
    fn gpu_fork_matches_original() {
        let n_features = 5;
        let (train_x, train_y) = make_data(150, n_features, 42);
        let params = RandomForestParameters::default()
            .with_n_estimators(10)
            .with_seed(10);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let gpu_forest = GpuForest::from_flat_forest(&flat, 100).unwrap();
        let forked = gpu_forest.fork(100);

        let (test_x, _) = make_data(50, n_features, 99);
        let test_x_f32 = test_x.mapv(|v| v as f32);

        let preds_orig = gpu_forest.predict(&test_x_f32.view()).unwrap();
        let preds_fork = forked.predict(&test_x_f32.view()).unwrap();

        assert_eq!(preds_orig.len(), preds_fork.len());
        for (i, (p1, p2)) in preds_orig.iter().zip(preds_fork.iter()).enumerate() {
            assert_eq!(p1, p2, "sample {i}: original={p1}, fork={p2}");
        }
    }

    // --- Test #12: n_samples > max_samples panics ---
    /// predict_submit must panic when the batch exceeds the pre-allocated capacity.
    #[test]
    #[should_panic(expected = "exceeds max_samples")]
    fn gpu_exceeds_max_samples_panics() {
        let n_features = 4;
        let (train_x, train_y) = make_data(100, n_features, 42);
        let params = RandomForestParameters::default()
            .with_n_estimators(5)
            .with_seed(12);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let gpu_forest = GpuForest::from_flat_forest(&flat, 10).unwrap(); // max_samples=10

        let (test_x, _) = make_data(20, n_features, 99); // 20 > 10
        let test_x_f32 = test_x.mapv(|v| v as f32);
        let _ = gpu_forest.predict(&test_x_f32.view()); // should panic
    }

    // --- Test #13: GPU predictions are deterministic across calls ---
    /// Three consecutive predict calls on the same GpuForest must return bit-for-bit
    /// identical results. Guards against non-determinism in the reduce pass.
    #[test]
    fn gpu_deterministic_across_calls() {
        let n_features = 5;
        let (train_x, train_y) = make_data(100, n_features, 42);
        let params = RandomForestParameters::default()
            .with_n_estimators(10)
            .with_seed(13);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let gpu_forest = GpuForest::from_flat_forest(&flat, 50).unwrap();

        let (test_x, _) = make_data(30, n_features, 99);
        let test_x_f32 = test_x.mapv(|v| v as f32);

        let preds1 = gpu_forest.predict(&test_x_f32.view()).unwrap();
        let preds2 = gpu_forest.predict(&test_x_f32.view()).unwrap();
        let preds3 = gpu_forest.predict(&test_x_f32.view()).unwrap();

        for i in 0..preds1.len() {
            assert_eq!(preds1[i], preds2[i], "run 1 vs 2, sample {i}");
            assert_eq!(preds1[i], preds3[i], "run 1 vs 3, sample {i}");
        }
    }

    // --- Test #11: predict_submit + collect pipelining ---
    /// Submit GPU work for two forked handles before collecting either, then verify
    /// both results are correct. This is the primary use case for the split
    /// predict_submit/collect API: letting the GPU work on both batches concurrently.
    #[test]
    fn gpu_pipelined_submit_collect() {
        let n_features = 6;
        let (train_x, train_y) = make_data(150, n_features, 42);

        let params = RandomForestParameters::default()
            .with_n_estimators(20)
            .with_seed(11);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);

        // Two independent handles sharing compiled pipelines and node data.
        let forest_a = GpuForest::from_flat_forest(&flat, 100).unwrap();
        let forest_b = forest_a.fork(100);

        let (test_x_a, _) = make_data(70, n_features, 101);
        let (test_x_b, _) = make_data(50, n_features, 202);
        let test_a_f32 = test_x_a.mapv(|v| v as f32);
        let test_b_f32 = test_x_b.mapv(|v| v as f32);

        // CPU baseline for both batches.
        let cpu_a = flat.predict(&test_a_f32.view());
        let cpu_b = flat.predict(&test_b_f32.view());

        // Submit both batches to the GPU without waiting for either to finish.
        let handle_a = forest_a
            .predict_submit(&test_a_f32.view())
            .unwrap()
            .unwrap();
        let handle_b = forest_b
            .predict_submit(&test_b_f32.view())
            .unwrap()
            .unwrap();

        // Collect in submission order; both results must be correct.
        let gpu_a = handle_a.collect().unwrap();
        let gpu_b = handle_b.collect().unwrap();

        assert_eq!(gpu_a.len(), 70);
        assert_eq!(gpu_b.len(), 50);

        for (i, (cpu, gpu)) in cpu_a.iter().zip(gpu_a.iter()).enumerate() {
            assert!(
                ((*cpu as f32) - gpu).abs() < 1e-5,
                "batch A sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
        for (i, (cpu, gpu)) in cpu_b.iter().zip(gpu_b.iter()).enumerate() {
            assert!(
                ((*cpu as f32) - gpu).abs() < 1e-5,
                "batch B sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
    }
}
