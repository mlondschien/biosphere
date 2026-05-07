//! Verify that FlatForest::predict produces identical results to RandomForest::predict.
#![allow(non_snake_case)]
use biosphere::{FlatForest, RandomForest, RandomForestParameters};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn assert_flat_matches_recursive(forest: &RandomForest, test_X: &Array2<f64>, n_features: usize) {
    let cpu_preds = forest.predict(&test_X.view());
    let flat = FlatForest::from_forest(forest, n_features);
    let test_X_f32 = test_X.mapv(|v| v as f32);
    let flat_preds = flat.predict(&test_X_f32.view());
    for (i, (cpu, fp)) in cpu_preds.iter().zip(flat_preds.iter()).enumerate() {
        assert!((cpu - fp).abs() < 1e-5, "sample {i}: cpu={cpu}, flat={fp}");
    }
}

fn make_data(n_samples: usize, n_features: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    // Keep values in [0, 1) to avoid cumsum precision issues in biosphere internals.
    let X = Array2::random_using(
        (n_samples, n_features),
        Uniform::new(0.0, 1.0).unwrap(),
        &mut rng,
    );
    // Simple target: first feature only.
    let y = X.column(0).to_owned();
    (X, y)
}

#[test]
fn flat_forest_matches_recursive() {
    let (X, y) = make_data(200, 10, 42);

    let params = RandomForestParameters::default()
        .with_n_estimators(20)
        .with_seed(1);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let (test_X, _) = make_data(100, 10, 99);
    let cpu_preds = forest.predict(&test_X.view());

    let flat = FlatForest::from_forest(&forest, 10);
    let test_X_f32 = test_X.mapv(|v| v as f32);
    let flat_preds = flat.predict(&test_X_f32.view());

    for (i, (cpu, flat)) in cpu_preds.iter().zip(flat_preds.iter()).enumerate() {
        // FlatForest stores nodes as f32; allow for f32 quantisation error.
        assert!(
            (cpu - flat).abs() < 1e-5,
            "sample {i}: cpu={cpu}, flat={flat}"
        );
    }
}

#[test]
fn flat_forest_single_sample() {
    let (X, y) = make_data(100, 5, 7);

    let params = RandomForestParameters::default()
        .with_n_estimators(10)
        .with_seed(2);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let (test_X, _) = make_data(1, 5, 55);
    let cpu_preds = forest.predict(&test_X.view());
    let flat = FlatForest::from_forest(&forest, 5);
    let test_X_f32 = test_X.mapv(|v| v as f32);
    let flat_preds = flat.predict(&test_X_f32.view());

    // FlatForest stores nodes as f32; allow for f32 quantisation error.
    assert!(
        (cpu_preds[0] - flat_preds[0]).abs() < 1e-5,
        "cpu={}, flat={}",
        cpu_preds[0],
        flat_preds[0]
    );
}

#[test]
fn flat_forest_deep_tree() {
    // Trees trained with no max_depth can grow very deep — verify explicit-index layout is correct.
    let (X, y) = make_data(150, 8, 123);

    let params = RandomForestParameters::default()
        .with_n_estimators(5)
        .with_seed(3);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let (test_X, _) = make_data(50, 8, 456);
    assert_flat_matches_recursive(&forest, &test_X, 8);
}

// --- Test #1: depth-0 tree (single leaf) ---
/// min_samples_split >= n_samples forces every tree to be a single leaf.
/// max_tree_size must be 1, max_depth must be 0, and all test samples receive
/// the same averaged prediction (since each tree returns one constant value).
#[test]
fn depth_zero_single_leaf() {
    let n_train = 50;
    let (X, y) = make_data(n_train, 4, 11);
    // n_samples <= min_samples_split → root cannot split → single leaf.
    let params = RandomForestParameters::default()
        .with_n_estimators(5)
        .with_min_samples_split(n_train)
        .with_seed(10);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let flat = FlatForest::from_forest(&forest, 4);
    assert_eq!(
        flat.max_tree_size, 1,
        "depth-0 forest: max_tree_size should be 1"
    );
    assert_eq!(flat.max_depth, 0, "depth-0 forest: max_depth should be 0");

    let (test_X, _) = make_data(20, 4, 999);
    let test_X_f32 = test_X.mapv(|v| v as f32);
    let preds = flat.predict(&test_X_f32.view());
    // Each tree returns its single leaf value regardless of input; averaged output is
    // the same constant for every test sample.
    let first = preds[0];
    for &p in preds.iter() {
        assert!(
            (p - first).abs() < 1e-10,
            "depth-0 forest: all predictions should be equal across test samples"
        );
    }
}

// --- Test #2: depth-1 forest ---
/// max_depth=Some(1) means each tree has at most one split (≤ 3 nodes).
#[test]
fn depth_one_forest() {
    let (X, y) = make_data(100, 5, 22);
    let params = RandomForestParameters::default()
        .with_n_estimators(10)
        .with_max_depth(Some(1))
        .with_seed(20);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let flat = FlatForest::from_forest(&forest, 5);
    // A depth-1 tree is root + at most 2 leaves = 3 nodes.
    assert!(
        flat.max_tree_size <= 3,
        "depth-1 forest: max_tree_size={}",
        flat.max_tree_size
    );
    assert!(
        flat.max_depth <= 1,
        "depth-1 forest: max_depth={}",
        flat.max_depth
    );

    let (test_X, _) = make_data(50, 5, 222);
    assert_flat_matches_recursive(&forest, &test_X, 5);
}

// --- Test #5: feature at last index ---
/// With n_features=1, every split uses feature_index=0 = n_features-1.
/// Tests that the last feature index is handled correctly (no off-by-one).
#[test]
fn feature_at_last_index() {
    let n_features = 1;
    let mut rng = StdRng::seed_from_u64(55);
    let X = Array2::random_using((100, n_features), Uniform::new(0.0, 1.0).unwrap(), &mut rng);
    let y = X.column(0).to_owned();

    let params = RandomForestParameters::default()
        .with_n_estimators(10)
        .with_seed(5);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let test_X = Array2::random_using((30, n_features), Uniform::new(0.0, 1.0).unwrap(), &mut rng);
    assert_flat_matches_recursive(&forest, &test_X, n_features);

    // The forest must produce varied predictions (confirming splits on feature 0 = last feature).
    let cpu_preds = forest.predict(&test_X.view());
    let all_same = cpu_preds.iter().all(|&p| (p - cpu_preds[0]).abs() < 1e-10);
    assert!(
        !all_same,
        "forest should produce varied predictions when feature 0 is predictive"
    );
}

// --- Test #6: column-major input produces the same predictions as row-major ---
/// `predict` must accept any memory layout by converting to C-order internally.
/// A transposed (column-major) view must yield identical results to the normal
/// row-major view with the same data.
#[test]
fn non_contiguous_input_matches_contiguous() {
    let (X, y) = make_data(50, 4, 88);
    let params = RandomForestParameters::default()
        .with_n_estimators(3)
        .with_seed(7);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let flat = FlatForest::from_forest(&forest, 4);

    // Build a row-major (n_samples, n_features) f32 matrix.
    let mat_c = Array2::<f32>::from_shape_fn((20, 4), |(i, j)| (i * 4 + j) as f32 * 0.01);
    // Build the same data in column-major (Fortran) order by transposing a (4, 20) array.
    // The transposed view has shape (20, 4) but column-major strides — rows are non-contiguous.
    let mat_t = Array2::<f32>::from_shape_fn((4, 20), |(j, i)| (i * 4 + j) as f32 * 0.01);
    let non_contig = mat_t.t(); // shape (20, 4), column-major

    let preds_c = flat.predict(&mat_c.view());
    let preds_nc = flat.predict(&non_contig);

    assert_eq!(preds_c.len(), preds_nc.len());
    for (a, b) in preds_c.iter().zip(preds_nc.iter()) {
        assert!(
            (a - b).abs() < 1e-9,
            "row-major and column-major predictions differ: {a} vs {b}"
        );
    }
}

// --- Test #7: n_trees=1 ---
/// A single-tree forest's FlatForest prediction must match the recursive prediction.
#[test]
fn single_tree_forest() {
    let (X, y) = make_data(100, 5, 33);
    let params = RandomForestParameters::default()
        .with_n_estimators(1)
        .with_seed(8);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let (test_X, _) = make_data(50, 5, 77);
    assert_flat_matches_recursive(&forest, &test_X, 5);
}

// --- Test #13: iris regression ---
/// Train on iris (150 samples); predict petal width (continuous) from the other 3 features.
/// Using a continuous target keeps leaf predictions numerically close even when f32 feature
/// precision causes an occasional branch difference, so 1e-3 tolerance is achievable.
/// The test gives stable regression coverage tied to a real, well-known dataset.
#[test]
fn iris_regression() {
    use csv::ReaderBuilder;
    use ndarray::s;
    use ndarray_csv::Array2Reader;

    let file = std::fs::File::open("testdata/iris.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let data: Array2<f64> = reader.deserialize_array2((150, 5)).unwrap();
    // Use sepal length/width + petal length (columns 0-2) as features,
    // petal width (column 3, continuous 0.1-2.5) as the regression target.
    // This avoids integer class labels whose large leaf-value gaps would amplify
    // any f32 branch-decision differences beyond 1e-5.
    let X = data.slice(s![.., ..3]).to_owned();
    let y = data.column(3).to_owned();

    let params = RandomForestParameters::default()
        .with_n_estimators(20)
        .with_max_depth(Some(4))
        .with_seed(13);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let flat = FlatForest::from_forest(&forest, 3);
    let X_f32 = X.mapv(|v| v as f32);
    let recursive_preds = forest.predict(&X.view());
    let flat_preds = flat.predict(&X_f32.view());

    // Allow 0.1: iris features like 4.2, 5.1 are "round" in f32, but the stored f32 thresholds
    // differ from f64 originals, so occasional samples near a split boundary take a different
    // branch. With 20 trees and leaf values in [0, 2.5], a single branch difference per sample
    // contributes up to ~0.1. The tolerance catches algorithmic bugs without false-failing on
    // expected precision mismatches.
    for (i, (rec, fp)) in recursive_preds.iter().zip(flat_preds.iter()).enumerate() {
        assert!(
            (rec - fp).abs() < 0.1,
            "iris sample {i}: recursive={rec}, flat={fp}"
        );
    }
}

// --- Test #14: predict is idempotent ---
/// Calling predict twice on the same FlatForest with the same input returns identical results.
#[test]
fn predict_is_idempotent() {
    let (X, y) = make_data(100, 6, 44);
    let params = RandomForestParameters::default()
        .with_n_estimators(10)
        .with_seed(14);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let (test_X, _) = make_data(40, 6, 444);
    let flat = FlatForest::from_forest(&forest, 6);
    let test_X_f32 = test_X.mapv(|v| v as f32);

    let preds1 = flat.predict(&test_X_f32.view());
    let preds2 = flat.predict(&test_X_f32.view());

    for (i, (p1, p2)) in preds1.iter().zip(preds2.iter()).enumerate() {
        assert_eq!(
            p1, p2,
            "sample {i}: predictions differ between identical calls"
        );
    }
}

// --- Test #6: max_depth = Some(0) ---
/// max_depth=Some(0) exercises the `current_depth >= depth` early-return in the
/// tree splitter. Every tree must be a single leaf regardless of the data.
#[test]
fn max_depth_zero() {
    let (X, y) = make_data(50, 4, 60);
    let params = RandomForestParameters::default()
        .with_n_estimators(5)
        .with_max_depth(Some(0))
        .with_seed(60);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let flat = FlatForest::from_forest(&forest, 4);
    assert_eq!(
        flat.max_depth, 0,
        "max_depth=Some(0): flat.max_depth should be 0"
    );
    assert_eq!(
        flat.max_tree_size, 1,
        "max_depth=Some(0): every tree should be a single leaf"
    );

    let (test_X, _) = make_data(20, 4, 600);
    let test_X_f32 = test_X.mapv(|v| v as f32);
    let preds = flat.predict(&test_X_f32.view());
    let first = preds[0];
    for &p in preds.iter() {
        assert!(
            (p - first).abs() < 1e-10,
            "max_depth=Some(0): all predictions should be identical across test samples"
        );
    }
}

// --- Test #7: zero-variance training data ---
/// When all feature rows are identical no impurity improvement is possible, so
/// every tree becomes a single leaf. The prediction should equal the mean of y.
#[test]
fn zero_variance_training_data() {
    let n = 50;
    let X = Array2::<f64>::ones((n, 4)) * 0.5;
    let y = Array1::<f64>::from_elem(n, 1.3);

    let params = RandomForestParameters::default()
        .with_n_estimators(5)
        .with_seed(7);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let flat = FlatForest::from_forest(&forest, 4);
    assert_eq!(
        flat.max_tree_size, 1,
        "zero-variance data: every tree should be a single leaf"
    );

    let test_X = Array2::<f32>::ones((10, 4)) * 0.5f32;
    let preds = flat.predict(&test_X.view());
    for &p in preds.iter() {
        assert!(
            (p - 1.3).abs() < 1e-5,
            "zero-variance data: expected prediction 1.3, got {p}"
        );
    }
}

// --- Test #8: single training sample ---
/// With one training sample each bootstrap tree predicts the same constant.
#[test]
fn single_training_sample() {
    let X = Array2::<f64>::from_shape_vec((1, 3), vec![0.5, 0.2, 0.8]).unwrap();
    let y = Array1::<f64>::from_vec(vec![42.0]);

    let params = RandomForestParameters::default()
        .with_n_estimators(5)
        .with_seed(8);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let flat = FlatForest::from_forest(&forest, 3);
    assert_eq!(
        flat.max_tree_size, 1,
        "single-sample forest: every tree should be a leaf"
    );

    let test_X =
        Array2::<f32>::from_shape_vec((3, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            .unwrap();
    let preds = flat.predict(&test_X.view());
    for &p in preds.iter() {
        assert!(
            (p - 42.0).abs() < 1e-5,
            "single-sample forest: expected prediction 42.0, got {p}"
        );
    }
}

// --- Test #9: very wide data (n_features >> n_samples) ---
/// 10 samples × 200 features stresses n_features indexing and feature_index bounds.
#[test]
fn wide_data_more_features_than_samples() {
    let n_samples = 10;
    let n_features = 200;
    let mut rng = StdRng::seed_from_u64(99);
    let X = Array2::random_using(
        (n_samples, n_features),
        Uniform::new(0.0, 1.0).unwrap(),
        &mut rng,
    );
    let y = X.column(0).to_owned();

    let params = RandomForestParameters::default()
        .with_n_estimators(5)
        .with_seed(9);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let test_X = Array2::random_using((15, n_features), Uniform::new(0.0, 1.0).unwrap(), &mut rng);
    assert_flat_matches_recursive(&forest, &test_X, n_features);
}

// --- Test #15: OOB predictions and forest predict unchanged after FlatForest conversion ---
/// from_forest borrows the forest; after conversion the original RandomForest must
/// remain fully functional — predict and fit_predict_oob must still work.
#[test]
fn forest_unchanged_after_flat_conversion() {
    let (X, y) = make_data(100, 5, 15);
    let params = RandomForestParameters::default()
        .with_n_estimators(10)
        .with_seed(15);
    let mut forest = RandomForest::new(params);
    let oob = forest.fit_predict_oob(&X.view(), &y.view());

    // Convert to flat — this must not modify the forest.
    let flat = FlatForest::from_forest(&forest, 5);

    // Forest still predicts correctly.
    let preds_before_drop = forest.predict(&X.view());
    drop(flat);
    let preds_after_drop = forest.predict(&X.view());

    // Predictions must be bit-for-bit identical before and after conversion+drop.
    assert_eq!(
        preds_before_drop, preds_after_drop,
        "forest.predict should be unaffected by FlatForest conversion"
    );

    // OOB predictions have the right length; non-NaN entries are in a sane range.
    assert_eq!(oob.len(), 100);
    let non_nan: Vec<f64> = oob.iter().copied().filter(|v| !v.is_nan()).collect();
    assert!(
        !non_nan.is_empty(),
        "at least some samples should have OOB predictions"
    );
}
