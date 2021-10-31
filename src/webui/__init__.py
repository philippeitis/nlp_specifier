import os

import requests
import streamlit as st

SERVICE_URL = os.getenv("NLP_SERVICE_URL", "http://0.0.0.0:5000")
st.title("")


def render_items(method, params):
    body = requests.get(
        f"{SERVICE_URL}/render/{method}",
        params=params,
    ).text

    body = "\n".join(line for line in body.split("\n") if line.strip())
    style = "<style>mark.entity { display: inline-block }</style>"

    st.write(f"{style}{body}", unsafe_allow_html=True)


def read_tokenization_params():
    sentence = st.text_input("Text to tokenize", "Returns true if and only if `x == 2`")

    methods = {
        "Parts of speech": "pos",
        "Dependencies": "deps",
        "Named entities": "ner",
        "Semantic role labels": "srl",
    }

    method = methods[st.selectbox("Render", methods)]

    if method in {"pos", "deps"}:
        retokenize = st.checkbox("Apply retokenization", value=True)
        params = {"sentence": sentence, "retokenize": retokenize}
    else:
        params = {"sentence": sentence}

    return st.button("Submit"), method, params


submit, method, params = read_tokenization_params()
if submit:
    render_items(method, params)
