# Should do check for Intel based systems to use Intel MKL
sudo apt install python3-dev graphviz gfortran
pip install -U pip setuptools wheel
pip install -U spacy[cuda112]
pip install -U stanza
pip install -U fastapi uvicorn
python -m spacy download en_core_web_lg
cd nlp && pip install -r requirements.txt ; cd ..