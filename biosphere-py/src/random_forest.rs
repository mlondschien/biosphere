use biosphere::RandomForest as BioForest;
use biosphere::{RandomForestParameters, Mtry};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::{PyResult, Python};
use pyo3::proc_macro::{pyclass, pymethods};

#[pyclass]
#[repr(transparent)]
pub struct RandomForest {
    pub forest: BioForest,
}

#[pymethods]
impl RandomForest {
    #[new]
    #[args(
        n_trees = 100,
        max_depth = 4,
        mtry = "None",
        min_samples_split = 2,
        min_samples_leaf = 1,
        seed = 0,
        n_jobs = "None"
    )]
    pub fn __init__(
        n_trees: usize,
        max_depth: Option<usize>,
        mtry: Mtry,
        min_samples_split: usize,
        min_samples_leaf: usize,
        seed: u64,
        n_jobs: Option<usize>,
    ) -> PyResult<Self> {
        let random_forest_parameters = RandomForestParameters::new(
            n_trees,
            seed,
            max_depth,
            mtry,
            min_samples_split,
            min_samples_leaf,
            n_jobs,
        );
        Ok(RandomForest {
            forest: BioForest::new(random_forest_parameters),
        })
    }

    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) {
        let X_array = X.as_array();
        let y_array = y.as_array();

        self.forest.fit(&X_array, &y_array);
    }

    #[allow(non_snake_case)]
    pub fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> &'py PyArray1<f64> {
        let X_array = X.as_array();
        self.forest.predict(&X_array).to_pyarray(py)
    }

    #[allow(non_snake_case)]
    pub fn fit_predict_oob<'py>(
        &mut self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<f64> {
        let X_array = X.as_array();
        let y_array = y.as_array();
        self.forest
            .fit_predict_oob(&X_array, &y_array)
            .to_pyarray(py)
    }
}
