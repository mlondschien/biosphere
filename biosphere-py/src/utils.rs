use biosphere::MaxFeatures;
use pyo3::exceptions;
use pyo3::prelude::{pyclass, FromPyObject, PyAny, PyErr, PyResult};

#[pyclass(name = "MaxFeatures")]
pub struct PyMaxFeatures {
    pub value: MaxFeatures,
}

impl PyMaxFeatures {
    pub fn default() -> Self {
        PyMaxFeatures {
            value: MaxFeatures::default(),
        }
    }
}

impl FromPyObject<'_> for PyMaxFeatures {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        if let Ok(value) = ob.extract::<usize>() {
            Ok(PyMaxFeatures {
                value: MaxFeatures::Value(value),
            })
        } else if let Ok(value) = ob.extract::<f64>() {
            Ok(PyMaxFeatures {
                value: MaxFeatures::Fraction(value),
            })
        } else if let Ok(value) = ob.extract::<Option<String>>() {
            if value.is_none() {
                Ok(PyMaxFeatures {
                    value: MaxFeatures::None,
                })
            } else if let Some(value) = value {
                if value == "sqrt" {
                    Ok(PyMaxFeatures {
                        value: MaxFeatures::Sqrt,
                    })
                } else {
                    Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                        "Unknown value for max_features: {}",
                        ob
                    )))
                }
            } else {
                Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                    "Unknown value for max_features: {}",
                    ob
                )))
            }
        } else {
            Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                "Unknown value for max_features: {}",
                ob
            )))
        }
    }
}
