use biosphere::DecisionTree as BioDecisionTree;
use biosphere::DecisionTreeParameters;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::{PyResult, Python};
use crate::utils::PyMaxFeatures;
use pyo3::{pyclass, pymethods, Bound};

#[pyclass]
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
}
