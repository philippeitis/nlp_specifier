pip install lxml
pip install maturin
maturin build --release -m ./Cargo.toml
pip install .