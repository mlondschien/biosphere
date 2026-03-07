use csv::ReaderBuilder;
use ndarray::{Array2, ArrayBase, Data, Ix1};
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
    let mut values: Vec<f64> = Vec::with_capacity(150 * 5);
    for record in reader.records() {
        for field in record.unwrap().iter() {
            values.push(field.parse::<f64>().unwrap());
        }
    }
    Array2::from_shape_vec((150, 5), values).unwrap()
}
