import logging
import time
from contextlib import contextmanager
from http import HTTPStatus
from io import BytesIO
from typing import List

import spacy
import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from tokenizer import SpacyModel, Tokenizer

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


TOKENIZE_OUT = {
    int(HTTPStatus.OK): {
        "description": "Tokens, tags, lemmas, and vector corresponding to input sentences",
        "content": {"application/msgpack": {}},
    }
}


class TokenizeIn(BaseModel):
    model: SpacyModel
    sentences: List[str]


@app.get("/tokenize", responses=TOKENIZE_OUT, response_class=Response)
def tokenize(request: TokenizeIn):
    model = request.model
    sentences = request.sentences
    with timer("Opening model took {elapsed:.5f}s"):
        tokenizer = Tokenizer.from_cache(f"./cache/{model}.spacy", model)

    with timer("Tokenization took {elapsed:.5f}s"):
        sentences = tokenizer.stokenize(sentences)

    with timer("Serialization took {elapsed:.5f}s"):
        response = BytesIO()
        response.write(b"\x81")
        response.write((0x5 << 5 | len("sentences")).to_bytes(1, byteorder="big"))
        response.write(b"sentences")
        write_array_len(response, len(sentences))

        for sentence in sentences:
            response.write(sentence.msgpack)
        response.seek(0)

    return StreamingResponse(response, media_type="application/msgpack")


@app.post("/persist_cache", responses={int(HTTPStatus.NO_CONTENT): {}})
def persist_cache():
    with timer("Persisting cache took {elapsed:.5f}s"):
        for model in Tokenizer.TOKEN_CACHE.keys():
            Tokenizer(model).write_data(f"./cache/{model}.spacy")
    return Response(status_code=HTTPStatus.NO_CONTENT)


class Explain(BaseModel):
    explanation: str


@app.get("/explain", response_model=Explain)
def explain(q: str):
    return {"explanation": spacy.explain(q)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
