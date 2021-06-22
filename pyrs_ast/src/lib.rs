use syn::{parse_file, Item};
use syn::File;
use syn_serde::json;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn ast_from_str(s: String) -> PyResult<String> {
    match parse_file(&s) {
        Ok(x) => {
            return PyResult::Ok(json::to_string_pretty(&x));
        },
        Err(e) => PyResult::Ok(e.to_string()),
    }
}

#[pymodule]
fn astx(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ast_from_str, m)?)?;

    Ok(())
}