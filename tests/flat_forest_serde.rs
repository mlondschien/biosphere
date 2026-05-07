//! Round-trip tests: RandomForest → FlatForest → postcard bytes → FlatForest,
//! verifying that predictions match the original RandomForest within f32 tolerance.
#![allow(non_snake_case)]
use biosphere::{FlatForest, MaxFeatures, RandomForest, RandomForestParameters};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn make_data(n_samples: usize, n_features: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let X = Array2::random_using(
        (n_samples, n_features),
        Uniform::new(0.0, 1.0).unwrap(),
        &mut rng,
    );
    let y = X.column(0).to_owned();
    (X, y)
}

fn round_trip(flat: &FlatForest) -> FlatForest {
    let bytes = postcard::to_stdvec(flat).expect("serialization failed");
    postcard::from_bytes(&bytes).expect("deserialization failed")
}

/// Assert that FlatForest predictions are within 1e-5 of RandomForest predictions
/// (f32 quantisation of leaf values).
fn assert_predictions_match(forest: &RandomForest, flat: &FlatForest, X: &Array2<f64>) {
    let rf_preds = forest.predict(&X.view());
    let X_f32 = X.mapv(|v| v as f32);
    let flat_preds = flat.predict(&X_f32.view());
    for (i, (rf, fp)) in rf_preds.iter().zip(flat_preds.iter()).enumerate() {
        assert!(
            (rf - fp).abs() < 1e-5,
            "sample {i}: RandomForest={rf}, FlatForest={fp}"
        );
    }
}

/// Large forest: 50 trees, 15 features, 150 training samples.
/// Verifies that the round-tripped FlatForest produces the same predictions
/// as the original RandomForest on a held-out test set.
#[test]
fn round_trip_large_forest() {
    let n_features = 15;
    let (X_train, y_train) = make_data(150, n_features, 1);
    let (X_test, _) = make_data(80, n_features, 2);

    let params = RandomForestParameters::default()
        .with_n_estimators(50)
        .with_seed(1);
    let mut forest = RandomForest::new(params);
    forest.fit(&X_train.view(), &y_train.view());

    let flat = FlatForest::from_forest(&forest, n_features);
    let restored = round_trip(&flat);

    assert_predictions_match(&forest, &restored, &X_test);
}

/// Same training set as inference set — tests that in-sample predictions survive
/// round-trip (exercises all leaves that were actually reached during training).
#[test]
fn round_trip_in_sample_predictions() {
    let n_features = 10;
    let (X, y) = make_data(120, n_features, 3);

    let params = RandomForestParameters::default()
        .with_n_estimators(30)
        .with_seed(3);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let flat = FlatForest::from_forest(&forest, n_features);
    let restored = round_trip(&flat);

    assert_predictions_match(&forest, &restored, &X);
}

/// Forest with capped depth — trees are shallow and uniform, so most trees
/// have the same node count and there is little padding to strip.
#[test]
fn round_trip_shallow_trees() {
    let n_features = 8;
    let (X_train, y_train) = make_data(150, n_features, 4);
    let (X_test, _) = make_data(50, n_features, 5);

    let params = RandomForestParameters::default()
        .with_n_estimators(40)
        .with_max_depth(Some(4))
        .with_seed(4);
    let mut forest = RandomForest::new(params);
    forest.fit(&X_train.view(), &y_train.view());

    let flat = FlatForest::from_forest(&forest, n_features);
    let restored = round_trip(&flat);

    assert_predictions_match(&forest, &restored, &X_test);
}

/// Forest with many features and sqrt feature sampling — exercises the
/// MaxFeatures::Sqrt path and a high-dimensional node layout.
#[test]
fn round_trip_sqrt_max_features() {
    let n_features = 20;
    let (X_train, y_train) = make_data(150, n_features, 6);
    let (X_test, _) = make_data(60, n_features, 7);

    let params = RandomForestParameters::default()
        .with_n_estimators(25)
        .with_max_features(MaxFeatures::Sqrt)
        .with_seed(6);
    let mut forest = RandomForest::new(params);
    forest.fit(&X_train.view(), &y_train.view());

    let flat = FlatForest::from_forest(&forest, n_features);
    let restored = round_trip(&flat);

    assert_predictions_match(&forest, &restored, &X_test);
}

/// Compact bytes are smaller than the raw padded node array, even before
/// any external compression.
#[test]
fn serialized_size_smaller_than_padded() {
    let n_features = 10;
    let (X, y) = make_data(150, n_features, 8);

    let params = RandomForestParameters::default()
        .with_n_estimators(50)
        .with_seed(8);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let flat = FlatForest::from_forest(&forest, n_features);
    let bytes = postcard::to_stdvec(&flat).unwrap();
    let padded_bytes = flat.meta.n_trees as usize
        * flat.meta.max_tree_size as usize
        * size_of::<biosphere::FlatNode>();

    assert!(
        bytes.len() < padded_bytes,
        "compact rmp bytes ({}) should be smaller than raw padded layout ({})",
        bytes.len(),
        padded_bytes,
    );
}
