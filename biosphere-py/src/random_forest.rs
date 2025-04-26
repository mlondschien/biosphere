use crate::utils::PyMaxFeatures;
use biosphere::RandomForest as BioForest;
use biosphere::RandomForestParameters;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::{PyResult, Python};
use pyo3::{pyclass, pymethods, Bound};

#[pyclass]
#[repr(transparent)]
pub struct RandomForest {
    pub forest: BioForest,
}

#[pymethods]
impl RandomForest {
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        max_depth = 4,
        max_features = PyMaxFeatures::default(),
        min_samples_split = 2,
        min_samples_leaf = 1,
        random_state = 0,
        n_jobs = None
    ))]
    pub fn __init__(
        n_estimators: usize,
        max_depth: Option<usize>,
        max_features: PyMaxFeatures,
        min_samples_split: usize,
        min_samples_leaf: usize,
        random_state: u64,
        n_jobs: Option<i32>,
    ) -> PyResult<Self> {
        let random_forest_parameters = RandomForestParameters::new(
            n_estimators,
            random_state,
            max_depth,
            max_features.value,
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
    pub fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> Bound<'py, PyArray1<f64>>{
        let X_array = X.as_array();
        self.forest.predict(&X_array).to_pyarray(py)
    }

    #[allow(non_snake_case)]
    pub fn fit_predict_oob<'py>(
        &mut self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let X_array = X.as_array();
        let y_array = y.as_array();
        self.forest
            .fit_predict_oob(&X_array, &y_array)
            .to_pyarray(py)
    }
}
