use csv::ReaderBuilder;
use ndarray::{Array2, ArrayBase, Data, Ix1};
use ndarray_csv::Array2Reader;
use std::fs::File;

/// Check if input is sorted. Used for testing.
///
/// From https://stackoverflow.com/questions/51272571/how-do-i-check-if-a-slice-is-sorted.
pub fn is_sorted(data: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> bool {
    data.windows(2).into_iter().all(|x| x[0] <= x[1])
}

pub fn load_iris() -> Array2<f64> {
    let file = File::open("testdata/iris.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let data: Array2<f64> = reader.deserialize_array2((150, 5)).unwrap();
    data
}
