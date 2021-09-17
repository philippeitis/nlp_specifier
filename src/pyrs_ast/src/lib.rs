use syn::{parse_file, ImplItemMethod, ItemFn, ItemStruct};
use syn::Expr;

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

#[pyfunction]
fn parse_expr(s: String) -> PyResult<String> {
    let e: Result<Expr, syn::Error> = syn::parse_str(&s);
    match e {
        Ok(x) => {
            return PyResult::Ok(json::to_string_pretty(&x));
        },
        Err(e) => PyResult::Ok(e.to_string()),
    }
}

#[pyfunction]
fn parse_impl_method(s: String) -> PyResult<String> {
    let e: Result<ImplItemMethod, syn::Error> = syn::parse_str(&s);
    match e {
        Ok(x) => {
            return PyResult::Ok(json::to_string_pretty(&x));
        },
        Err(e) => PyResult::Ok(e.to_string()),
    }
}

#[pyfunction]
fn parse_fn(s: String) -> PyResult<String> {
    let e: Result<ItemFn, syn::Error> = syn::parse_str(&s);
    match e {
        Ok(x) => {
            return PyResult::Ok(json::to_string_pretty(&x));
        },
        Err(e) => PyResult::Ok(e.to_string()),
    }
}

#[pyfunction]
fn parse_struct(s: String) -> PyResult<String> {
    let e: Result<ItemStruct, syn::Error> = syn::parse_str(&s);
    match e {
        Ok(x) => {
            return PyResult::Ok(json::to_string_pretty(&x));
        },
        Err(e) => PyResult::Ok(e.to_string()),
    }
}

#[pymodule]
#[pyo3(name="astx")]
fn astx(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ast_from_str, m)?)?;
    m.add_function(wrap_pyfunction!(parse_expr, m)?)?;
    m.add_function(wrap_pyfunction!(parse_impl_method, m)?)?;
    m.add_function(wrap_pyfunction!(parse_fn, m)?)?;
    m.add_function(wrap_pyfunction!(parse_struct, m)?)?;

    Ok(())
}