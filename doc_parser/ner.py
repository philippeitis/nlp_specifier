import json
import requests
import unidecode

SRL_URL = "http://127.0.0.8:701/srl"
NER_URL = "http://127.0.0.8:702/ner"


class NLPError(ValueError):
    pass


def request_helper(url: str, msg: dict) -> dict:
    res = requests.post(url, json=msg)
    body = json.loads(res.content.decode("utf-8"))
    if not res.ok:
        raise NLPError(body.get("message", "Unknown server error."))

    if "message" in body and not body["success"]:
        raise NLPError(body["message"])

    return body


def ner_and_srl(text: str, srl_url=SRL_URL, ner_url=NER_URL) -> dict:
    msg = {"text": unidecode.unidecode(text)}
    return {**request_helper(ner_url, msg), **request_helper(srl_url, msg), **msg}


def print_ner_and_srl(text: str, srl_url=SRL_URL, ner_url=NER_URL):
    print(json.dumps(ner_and_srl(text, srl_url, ner_url), indent=4))


if __name__ == "__main__":
    print_ner_and_srl("Converts an integer from big endian to the target’s endianness.")
