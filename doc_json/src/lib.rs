use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;

use home::rustup_home as rustup_home_;

#[pyfunction]
fn rustup_home() -> PyResult<PathBuf> {
    match rustup_home_() {
        Ok(path) => PyResult::Ok(path),
        Err(e) => PyResult::Err(PyValueError::new_err(e.to_string())),
    }
}

#[pymodule]
#[pyo3(name="py_cargo_utils")]
fn py_cargo_utils(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rustup_home, m)?)?;

    Ok(())
}