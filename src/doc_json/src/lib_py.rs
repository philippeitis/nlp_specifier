use std::path::{PathBuf};
use home::rustup_home as rustup_home_;

use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use syn::{ImplItemMethod, ItemFn, ItemStruct};

use syn_serde::json;

use crate::{
    get_toolchain_dirs as get_toolchains_,
    find_all_files_in_root_dir as find_all_files_in_root_dir_,
    DocItem, parse_file,
};
use pyo3::types::{PyList, PyTuple};

fn parse_impl_method(s: String) -> String {
    let e: Result<ImplItemMethod, syn::Error> = syn::parse_str(&s);
    match e {
        Ok(x) => json::to_string_pretty(&x),
        Err(e) => e.to_string(),
    }
}

fn parse_fn(s: String) -> String {
    let e: Result<ItemFn, syn::Error> = syn::parse_str(&s);
    match e {
        Ok(x) => return json::to_string_pretty(&x),
        Err(e) => e.to_string(),
    }
}

fn parse_struct(s: String) -> String {
    let e: Result<ItemStruct, syn::Error> = syn::parse_str(&s);
    match e {
        Ok(x) => json::to_string_pretty(&x),
        Err(e) => e.to_string(),
    }
}

#[pyfunction]
fn rustup_home() -> PyResult<PathBuf> {
    rustup_home_().map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn get_toolchains() -> PyResult<Vec<PathBuf>> {
    get_toolchains_().map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn find_all_files_in_root_dir(path: String) -> PyResult<Vec<PathBuf>> {
    find_all_files_in_root_dir_(path).map_err(|e| PyValueError::new_err(e.to_string()))
}

enum ItemContainer {
    Fn(String),
    Struct(String, Vec<String>),
    Primitive(String, Vec<String>)
}

#[pyfunction]
fn parse_all_files(path: String) -> PyResult<PyObject> {
    let items: Vec<(PathBuf, ItemContainer)> = find_all_files_in_root_dir(path)?.into_par_iter()
        .map(|path|
                 parse_file(&path).map(|item| (path, match item {
                     DocItem::Fn(f) => ItemContainer::Fn(parse_fn(f.s)),
                     DocItem::Struct(s) => {
                         ItemContainer::Struct(parse_struct(s.s), s.methods.into_iter().map(parse_impl_method).collect())
                     },
                     DocItem::Primitive(s) => {
                         ItemContainer::Primitive(parse_struct(s.s), s.methods.into_iter().map(parse_impl_method).collect())
                     }
                 }))
        )
        .filter_map(Result::ok)
        .collect();

    Python::with_gil(|py| {
        let py_output = PyList::empty(py);
        for (path, val) in items {
            let path = path.to_object(py);
            match val {
                ItemContainer::Fn(s) => {
                    let s = s.to_object(py);
                    let name = "fn".to_object(py);
                    py_output.append(PyTuple::new(py, [name, path, s]))?;
                }
                ItemContainer::Struct(s, methods) => {
                    let s = s.to_object(py);
                    let name = "struct".to_object(py);
                    let methods = methods.to_object(py);
                    py_output.append(PyTuple::new(py, [name, path, s, methods]))?;
                }
                ItemContainer::Primitive(s, methods) => {
                    let s = s.to_object(py);
                    let name = "primitive".to_object(py);
                    let methods = methods.to_object(py);
                    py_output.append(PyTuple::new(py, [name, path, s, methods]))?;
                }
            }
        }
        Ok(py_output.to_object(py))
    })
}


#[pymodule]
#[pyo3(name = "py_cargo_utils")]
fn py_cargo_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rustup_home, m)?)?;
    m.add_function(wrap_pyfunction!(get_toolchains, m)?)?;
    m.add_function(wrap_pyfunction!(find_all_files_in_root_dir, m)?)?;
    m.add_function(wrap_pyfunction!(parse_all_files, m)?)?;

    Ok(())
}
