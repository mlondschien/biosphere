mod decision_tree;
mod random_forest;

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn biosphere(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<decision_tree::DecisionTree>()?;
    m.add_class::<random_forest::RandomForest>()?;
    Ok(())
}
