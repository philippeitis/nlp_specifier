import json
from os import getenv

import requests
import unidecode

SRL_URL = getenv("SRL_SERVICE_URL", "http://127.0.0.8:701/srl")
NER_URL = getenv("NER_SERVICE_URL", "http://127.0.0.8:702/ner")


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
    sentences = [
        "Removes and returns the element at position index within the vector, shifting all elements after it to the left.",
        "Removes the last element from a vector and returns it, or None if it is empty.",
        "whichptr indicates which xbuffer holds the final iMCU row.",
    ]
    # "Removes [element index] vector"
    # Remove `index` `self`
    # "Removes [last element] vector" <- last element?

    # "shifting all elements after it to the left.",
    for sentence in sentences:
        print_ner_and_srl(sentence)

    # Passes:
    # Type inference (identify possible blobs in things like Ret)
    # Function resolution (eg. find functions with matching signatures, find functions with overlapping keywords)
