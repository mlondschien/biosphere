use csv::ReaderBuilder;
use ndarray::{s, Array2, ArrayView2, Axis};
use ndarray_csv::Array2Reader;
use std::fs::File;

/// Check if input is sorted. Used for testing.
///
/// From https://stackoverflow.com/questions/51272571/how-do-i-check-if-a-slice-is-sorted.
pub fn is_sorted<T>(data: &[T]) -> bool
where
    T: std::cmp::PartialOrd,
{
    data.windows(2).all(|w| w[0] <= w[1])
}

pub fn arrange_samples(
    samples: &[usize],
    features: &[usize],
    X: &ArrayView2<f64>,
) -> Vec<Vec<usize>> {
    let mut samples_out: Vec<Vec<usize>> = vec![];
    for feature_idx in 0..features.len() {
        let mut sample = samples.to_vec();
        sample.sort_by(|&idx1, &idx2| {
            X[[idx1, feature_idx]]
                .partial_cmp(&X[[idx2, feature_idx]])
                .unwrap()
        });
        assert!(is_sorted(
            X.slice(s![.., feature_idx])
                .select(Axis(0), &sample)
                .as_slice()
                .unwrap()
        ));
        samples_out.push(sample);
    }
    samples_out
}

pub fn load_iris() -> Array2<f64> {
    let file = File::open("testdata/iris.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let data: Array2<f64> = reader.deserialize_array2((150, 5)).unwrap();
    data
}
