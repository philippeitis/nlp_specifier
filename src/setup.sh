sudo apt install python3-dev python-dev graphviz
pip install -U pip setuptools wheel
pip install -U spacy[cuda112]
python -m spacy download en_core_web_lg
cd nlp && pip install -r requirements.txt ; cd ..