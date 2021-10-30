sudo apt install graphviz
pip install -U pip setuptools wheel
pip install -U stanza
pip install -r ./visualization/requirements.txt
pip install -r ./codegen/requirements.txt
pip install -r ./nlp/requirements.txt
python -m spacy download en_core_web_lg
