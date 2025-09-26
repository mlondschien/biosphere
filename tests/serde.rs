#[cfg(feature = "serde")]
use biosphere::{DecisionTreeParameters, MaxFeatures, RandomForest, RandomForestParameters};
#[cfg(feature = "serde")]
use ndarray::{array, ArrayView2, ArrayView1};
#[cfg(feature = "serde")]
use serde_json5;

#[cfg(feature = "serde")]
#[test]
fn decision_tree_parameters_roundtrip() {
    let params = DecisionTreeParameters::new(Some(3), MaxFeatures::Value(2), 4, 1, 42);
    let json = serde_json5::to_string(&params).unwrap();
    let other: DecisionTreeParameters = serde_json5::from_str(&json).unwrap();
    assert_eq!(params.max_depth, other.max_depth);
    match (params.max_features, other.max_features) {
        (MaxFeatures::None, MaxFeatures::None) => {},
        (MaxFeatures::Fraction(a), MaxFeatures::Fraction(b)) => assert_eq!(a, b),
        (MaxFeatures::Value(a), MaxFeatures::Value(b)) => assert_eq!(a, b),
        (MaxFeatures::Sqrt, MaxFeatures::Sqrt) => {},
        _ => panic!("max_features mismatch"),
    }
    assert_eq!(params.min_samples_split, other.min_samples_split);
    assert_eq!(params.min_samples_leaf, other.min_samples_leaf);
    assert_eq!(params.random_state, other.random_state);
}

#[cfg(feature = "serde")]
#[test]
fn random_forest_roundtrip() {
    let x = array![[0.], [1.], [2.], [3.], [4.]];
    let y = array![0., 1., 2., 3., 4.];
    let view_x: ArrayView2<f64> = x.view();
    let view_y: ArrayView1<f64> = y.view();
    let mut forest = RandomForest::new(
        RandomForestParameters::default()
            .with_n_estimators(1)
            .with_max_depth(Some(2))
            .with_seed(0),
    );
    forest.fit(&view_x, &view_y);
    let preds_before = forest.predict(&view_x);

    let json = serde_json5::to_string(&forest).unwrap();
    let deserialized: RandomForest = serde_json5::from_str(&json).unwrap();
    let preds_after = deserialized.predict(&view_x);

    assert_eq!(preds_before, preds_after);
}
