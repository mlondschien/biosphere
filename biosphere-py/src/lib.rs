use biosphere::RandomForest;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

// Note: This has to match the lib.name in Cargo.toml.
#[allow(non_snake_case)] // Allow capital X for arrays.
#[pymodule]
fn biosphere<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn predict_oob<'py>(
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        n_trees: Option<u16>,
        max_depth: Option<u16>,
        mtry: Option<u16>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        min_gain_to_split: Option<f64>,
        seed: Option<u64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let X_array = X.as_array();
        let y_array = y.as_array();
        let forest = RandomForest::new(
            &X_array,
            &y_array,
            n_trees,
            max_depth,
            mtry,
            min_samples_split,
            min_samples_leaf,
            min_gain_to_split,
            seed,
        );

        Ok(forest.predict().to_pyarray(py))
    }
    Ok(())
}
