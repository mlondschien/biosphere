use biosphere::MaxFeatures;
use pyo3::exceptions;
use pyo3::prelude::{pyclass, FromPyObject, PyAny, PyErr};
use pyo3::types::PyAnyMethods;

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

impl<'py> FromPyObject<'_, 'py> for PyMaxFeatures {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = ob.extract::<usize>() {
            Ok(PyMaxFeatures {
                value: MaxFeatures::Value(value),
            })
        } else if let Ok(value) = ob.extract::<f64>() {
            Ok(PyMaxFeatures {
                value: MaxFeatures::Fraction(value),
            })
        } else if ob.is_none() {
            Ok(PyMaxFeatures {
                value: MaxFeatures::None,
            })
        } else if let Ok(value) = ob.extract::<String>() {
            if value == "sqrt" {
                Ok(PyMaxFeatures {
                    value: MaxFeatures::Sqrt,
                })
            } else {
                Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                    "Unknown value for max_features: {:?}",
                    value
                )))
            }
        } else {
            Err(PyErr::new::<exceptions::PyTypeError, _>(
                "Unknown value for max_features (invalid type)".to_string()
            ))
        }
    }
}
