import time
from contextlib import contextmanager
from http import HTTPStatus
from io import BytesIO

import flask
import msgpack
import spacy
from flask import request, send_file
from marshmallow import Schema, fields
from tokenizer import Tokenizer

app = flask.Flask(__name__)
app.config["DEBUG"] = True


class TokenizerGet(Schema):
    model = fields.Str()
    sentences = fields.List(fields.String())


@app.route("/", methods=["GET"])
def main():
    return ""


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
    app.logger.info(fstring.format(elapsed=elapsed))


@app.route("/tokenize", methods=["GET"])
def tokenize():
    form_data = request.get_json()
    errors = TokenizerGet().validate(form_data)
    if errors:
        print(errors)
        return msgpack.packb({"error": "malformed input"}), HTTPStatus.BAD_REQUEST

    model = form_data["model"]

    with timer("Opening model took {elapsed:.5f}s"):
        tokenizer = Tokenizer.from_cache(f"./cache/{model}.spacy", model)

    with timer("Tokenization took {elapsed:.5f}s"):
        sentences = tokenizer.stokenize(form_data["sentences"])

    with timer("Serialization took {elapsed:.5f}s"):
        response = BytesIO()
        response.write(b"\x81")
        response.write((0x5 << 5 | len("sentences")).to_bytes(1, byteorder="big"))
        response.write(b"sentences")
        write_array_len(response, len(sentences))

        for sentence in sentences:
            response.write(sentence.msgpack)
        response.seek(0)

    return send_file(response, mimetype="application/msgpack"), HTTPStatus.OK


@app.route("/persist_cache", methods=["POST"])
def persist_cache():
    for model in Tokenizer.TOKEN_CACHE.keys():
        Tokenizer(model).write_data(f"./cache/{model}.spacy")
    return "", HTTPStatus.NO_CONTENT


@app.route("/explain", methods=["GET"])
def explain():
    target = request.args.get("q")
    if target is None:
        return (
            flask.jsonify({"error": "no value for 'q' specified"}),
            HTTPStatus.BAD_REQUEST,
        )
    return (
        flask.jsonify({"explanation": spacy.explain(target)}),
        HTTPStatus.OK,
    )


if __name__ == "__main__":
    app.run()
