from io import BytesIO

import flask
import msgpack
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


@app.route("/tokenize", methods=["GET"])
def tokenize():
    form_data = request.get_json()
    print(f"format_data: {form_data}")
    errors = TokenizerGet().validate(form_data)
    if errors:
        print(errors)
        return msgpack.packb({"error": "malformed input"}), 400

    model = form_data["model"]
    tokenizer = Tokenizer.from_cache(f"./cache/{model}.spacy", model)
    sentences = tokenizer.stokenize(form_data["sentences"])

    response = BytesIO()
    response.write(b"\x81")
    response.write((0x5 << 5 | len("sentences")).to_bytes(1, byteorder="big"))
    response.write(b"sentences")
    write_array_len(response, len(sentences))
    for sentence in sentences:
        response.write(sentence.msgpack())
    response.seek(0)
    return send_file(response, mimetype="application/msgpack"), 200


if __name__ == "__main__":
    app.run()
