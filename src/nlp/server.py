import logging
import time
from contextlib import contextmanager
from http import HTTPStatus
from io import BytesIO
from typing import List, Optional

import spacy
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel
from starlette.responses import JSONResponse, StreamingResponse
from tokenizer import SpacyModel, Tokenizer

REF_TEMPLATE = "#/components/schemas/{model}"
logger = logging.getLogger(__name__)
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
    model: SpacyModel
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


openapi = app.openapi()
app.openapi = custom_openapi
app.openapi_schema = None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
