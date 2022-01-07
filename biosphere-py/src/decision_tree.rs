use biosphere::tree::DecisionTree as BioDecisionTree;
use biosphere::tree::DecisionTreeParameters;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::{PyResult, Python};
use pyo3::proc_macro::{pyclass, pymethods};

#[pyclass]
#[repr(transparent)]
pub struct DecisionTree {
    pub tree: BioDecisionTree,
}

#[pymethods]
impl DecisionTree {
    #[new]
    #[args(
        max_depth = 4,
        mtry = "None",
        min_samples_split = 2,
        min_samples_leaf = 1,
        seed = 1
    )]
    pub fn __init__(
        max_depth: Option<usize>,
        mtry: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let decision_tree_parameters =
            DecisionTreeParameters::new(max_depth, mtry, min_samples_split, min_samples_leaf, seed);
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
    pub fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> &'py PyArray1<f64> {
        let X_array = X.as_array();
        self.tree.predict(&X_array).to_pyarray(py)
    }
}
