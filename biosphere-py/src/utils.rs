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

impl<'source> FromPyObject<'source> for PyMaxFeatures {
    fn extract_bound(ob: &pyo3::Bound<'source, PyAny>) -> PyResult<Self> {
        if let Ok(value) = usize::extract_bound(ob) {
            Ok(PyMaxFeatures {
                value: MaxFeatures::Value(value),
            })
        } else if let Ok(value) = f64::extract_bound(ob) {
            Ok(PyMaxFeatures {
                value: MaxFeatures::Fraction(value),
            })
        } else if let Ok(value) = Option::<String>::extract_bound(ob) {
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
                        "Unknown value for max_features: {:?}",
                        value
                    )))
                }
            } else {
                Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Unknown value for max_features (null option)".to_string()
                ))
            }
        } else {
            Err(PyErr::new::<exceptions::PyTypeError, _>(
                "Unknown value for max_features (invalid type)".to_string()
            ))
        }
    }
}
