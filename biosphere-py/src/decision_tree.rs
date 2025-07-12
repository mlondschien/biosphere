use biosphere::DecisionTree as BioDecisionTree;
use biosphere::DecisionTreeParameters;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::{PyResult, Python};
use crate::utils::PyMaxFeatures;
use pyo3::{pyclass, pymethods, Bound};
#[cfg(feature = "serde")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "serde")]
use serde_json5;

#[pyclass(module = "biosphere")]
#[repr(transparent)]
pub struct DecisionTree {
    pub tree: BioDecisionTree,
}

#[pymethods]
impl DecisionTree {
    #[new]
    #[pyo3(signature = (
        max_depth = 4,
        max_features = PyMaxFeatures::default(),
        min_samples_split = 2,
        min_samples_leaf = 1,
        random_state = 0
    ))]
    pub fn __init__(
        max_depth: Option<usize>,
        max_features: PyMaxFeatures,
        min_samples_split: usize,
        min_samples_leaf: usize,
        random_state: u64,
    ) -> PyResult<Self> {
        let decision_tree_parameters = DecisionTreeParameters::new(
            max_depth,
            max_features.value,
            min_samples_split,
            min_samples_leaf,
            random_state,
        );
        Ok(DecisionTree {
            tree: BioDecisionTree::new(decision_tree_parameters),
        })
    }

    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) {
        let X_array = X.as_array();
        let y_array = y.as_array();

        self.tree.fit(&X_array, &y_array);
    }

    #[allow(non_snake_case)]
    pub fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> Bound<'py, PyArray1<f64>> {
        let X_array = X.as_array();
        self.tree.predict(&X_array).to_pyarray(py)
    }

    #[cfg(feature = "serde")]
    #[pyo3(name = "__getstate__")]
    fn getstate(&self) -> PyResult<String> {
        serde_json5::to_string(&self.tree).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[cfg(feature = "serde")]
    #[pyo3(name = "__setstate__")]
    fn setstate(&mut self, state: &str) -> PyResult<()> {
        self.tree = serde_json5::from_str(state).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }
}
