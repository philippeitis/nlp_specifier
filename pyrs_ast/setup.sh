sudo apt install python3-dev python-dev
pip install maturin
maturin build --release
pip install .