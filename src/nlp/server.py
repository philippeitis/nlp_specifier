import logging
import time
from contextlib import contextmanager
from http import HTTPStatus
from io import BytesIO
from typing import List, Optional

import spacy
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel
from starlette.responses import JSONResponse, StreamingResponse
from tokenizer import SpacyModel, Tokenizer

REF_TEMPLATE = "#/components/schemas/{model}"
logger = logging.getLogger("specifiernlp")
logger.setLevel(logging.INFO)

app = FastAPI(debug=True)


def write_array_len(data: BytesIO, arr_len):
    if arr_len < 16:
        data.write((0x9 << 4 | arr_len).to_bytes(1, byteorder="big"))
    elif arr_len < (2 ** 16):
        data.write(b"\xdc")
        data.write(arr_len.to_bytes(2, byteorder="big"))
    elif arr_len < (2 ** 32):
        data.write(b"\xdd")
        data.write(arr_len.to_bytes(4, byteorder="big"))


@contextmanager
def timer(fstring):
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    logger.info(fstring.format(elapsed=elapsed))


class TokenizeIn(BaseModel):
    model: SpacyModel = SpacyModel.EN_SM
    sentences: List[str]


class Token(BaseModel):
    tag: str
    text: str
    lemma: str


class SentenceOut(BaseModel):
    text: str
    tokens: List[Token]
    vector: Optional[List[float]]


class TokenizeOut(BaseModel):
    sentences: List[SentenceOut]


TOKENIZE_OUT = {
    int(HTTPStatus.OK): {
        "description": "Tokens, tags, lemmas, and vector corresponding to input sentences",
        "content": {
            "application/msgpack": {},
            "application/json": {
                "schema": TokenizeOut.schema(ref_template=REF_TEMPLATE),
            },
        },
    }
}


@app.get("/tokenize", responses=TOKENIZE_OUT, response_class=Response)
def tokenize(
    request: TokenizeIn, accept: Optional[str] = Header(default="application/msgpack")
):
    model = request.model
    sentences = request.sentences

    if accept == "*/*":
        accept = "application/msgpack"

    if accept not in {"application/msgpack", "application/json"}:
        logger.error(f"Received bad header: {accept}")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Expected accept header application/msgpack, application/json, got {accept}",
        )

    with timer("Opening model took {elapsed:.5f}s"):
        tokenizer = Tokenizer.from_cache(f"./cache/{model}.spacy", model)

    with timer("Tokenization took {elapsed:.5f}s"):
        sentences = tokenizer.stokenize(sentences)

    with timer("Serialization took {elapsed:.5f}s"):
        if accept == "application/msgpack":
            response = BytesIO()
            response.write(b"\x81")
            response.write((0x5 << 5 | len("sentences")).to_bytes(1, byteorder="big"))
            response.write(b"sentences")
            write_array_len(response, len(sentences))

            for sentence in sentences:
                response.write(sentence.msgpack)
            response.seek(0)
            output = StreamingResponse(response, media_type="application/msgpack")
        else:
            output = JSONResponse(
                {"sentences": [sentence.json for sentence in sentences]},
                media_type="application/json",
            )
    return output


class Explain(BaseModel):
    explanation: Optional[str]


@app.get("/explain", response_model=Explain)
async def explain(q: str):
    return Explain(explanation=spacy.explain(q))


class Models(BaseModel):
    models: List[SpacyModel]


@app.get("/models", response_model=Models)
async def models(
    has_vec: Optional[bool] = Query(
        None, description="Filter for models which have word vectors"
    ),
    lang: str = "en",
):
    import requests
    from spacy import about
    from spacy.cli.validate import reformat_version
    from spacy.util import (get_installed_models, get_model_meta,
                            get_package_path, get_package_version,
                            is_compatible_version)

    r = requests.get(about.__compatibility__)

    if r.status_code != 200:
        return Response("", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    compat = r.json()["spacy"]
    all_models = set()
    installed_models = get_installed_models()
    for spacy_v, models in dict(compat).items():
        all_models.update(models.keys())
        for model, model_vs in models.items():
            compat[spacy_v][model] = [reformat_version(v) for v in model_vs]
    pkgs = {}

    for pkg_name in installed_models:
        package = pkg_name.replace("-", "_")
        version = get_package_version(pkg_name)
        if package in compat:
            is_compat = version in compat[package]
            spacy_version = about.__version__
            model_meta = {}
        else:
            model_path = get_package_path(package)
            model_meta = get_model_meta(model_path)
            spacy_version = model_meta.get("spacy_version", "n/a")
            is_compat = is_compatible_version(about.__version__, spacy_version)  # type: ignore[assignment]
        pkgs[pkg_name] = {
            "name": package,
            "version": version,
            "spacy": spacy_version,
            "compat": is_compat,
            "meta": model_meta,
        }

    matches = []
    for name, data in pkgs.items():
        if data["meta"].get("lang") == lang and data["compat"]:
            components = data["meta"].get("components")
            model_has_vec = components and "tok2vec" in components
            if has_vec is None or model_has_vec == has_vec:
                matches.append(name)

    return Models(models=matches)


@app.on_event("shutdown")
def shutdown():
    with timer("Persisting cache took {elapsed:.5f}s"):
        for model in Tokenizer.TOKEN_CACHE.keys():
            Tokenizer(model).write_data(f"./cache/{model}.spacy")


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    for component in (Token, SentenceOut):
        openapi["components"]["schemas"][component.__name__] = component.schema(
            ref_template=REF_TEMPLATE
        )

    app.openapi_schema = openapi
    return app.openapi_schema


def init_loggers():
    formatter = logging.Formatter("[%(name)s/%(funcName)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    logging.getLogger().setLevel(logging.WARNING)
    uvilog = logging.getLogger("uvicorn")
    uvilog.propagate = False


openapi = app.openapi()
app.openapi = custom_openapi
app.openapi_schema = None

if __name__ == "__main__":
    init_loggers()
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level=logging.INFO)
