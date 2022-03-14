use biosphere::Mtry;
use pyo3::exceptions;
use pyo3::prelude::{pyclass, FromPyObject, PyAny, PyErr, PyResult};

#[pyclass(name = "Mtry")]
pub struct PyMtry {
    pub mtry: Mtry,
}

impl PyMtry {
    pub fn default() -> Self {
        PyMtry {
            mtry: Mtry::default(),
        }
    }
}

impl FromPyObject<'_> for PyMtry {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        if let Ok(value) = ob.extract::<usize>() {
            Ok(PyMtry {
                mtry: Mtry::Value(value),
            })
        } else if let Ok(value) = ob.extract::<f64>() {
            Ok(PyMtry {
                mtry: Mtry::Fraction(value),
            })
        } else if let Ok(value) = ob.extract::<Option<String>>() {
            if value.is_none() {
                Ok(PyMtry { mtry: Mtry::None })
            } else if let Some(value) = value {
                if value == "sqrt" {
                    Ok(PyMtry { mtry: Mtry::Sqrt })
                } else {
                    Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                        "Unknown mtry: {}",
                        ob
                    )))
                }
            } else {
                Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                    "Unknown mtry: {}",
                    ob
                )))
            }
        } else {
            Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                "Unknown mtry: {}",
                ob
            )))
        }
    }
}
