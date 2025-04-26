mod decision_tree;
mod random_forest;
mod utils;
use pyo3::prelude::*;

#[pymodule]
fn biosphere(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<decision_tree::DecisionTree>()?;
    module.add_class::<random_forest::RandomForest>()?;
    
    Ok(())
}