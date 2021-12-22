use csv::ReaderBuilder;
use ndarray::{arr1, s, Array2};
use ndarray_csv::Array2Reader;
use oobforest::testing::arrange_samples;
use oobforest::DecisionTree;
use std::fs::File;

#[allow(non_snake_case)]
#[test]
fn test_integration_tree() {
    let file = File::open("testdata/iris.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let data: Array2<f64> = reader.deserialize_array2((150, 5)).unwrap();

    let X = data.slice(s![0..100, 0..4]).to_owned();
    let y = data.slice(s![0..100, 4]).to_owned();

    let mut oob_samples = (0..100).collect::<Vec<usize>>();
    let samples = arrange_samples(&oob_samples, &[0, 1, 2, 3], &X);

    let tree = DecisionTree {
        X: &X,
        y: &y,
        max_depth: 8,
        features: vec![0, 1, 2, 3],
    };
    let result = tree.split(samples, &mut oob_samples, vec![0, 1, 2, 3], 0);

    let mut predictions = arr1(&[0.0; 100]);
    for (idxs, val) in result.iter() {
        for idx in *idxs {
            predictions[*idx] = *val;
        }
    }

    assert_eq!(predictions, y);
}
