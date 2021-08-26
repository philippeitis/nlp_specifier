pip install lxml
pip install maturin
maturin build --release -m ./py_cargo_utils/Cargo.toml
pip install .