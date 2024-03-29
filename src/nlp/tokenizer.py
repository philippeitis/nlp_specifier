import io
import logging
import sys
from collections import defaultdict
from enum import Enum
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Union

import msgpack
import numpy as np
import spacy
import unidecode
from spacy.tokens import Doc, DocBin

try:
    from fix_tokens import fix_tokens
    from ner import ner_and_srl
except ModuleNotFoundError:
    from .fix_tokens import fix_tokens
    from .ner import ner_and_srl

LOGGER = logging.getLogger(__name__)

if not Doc.has_extension("raw_text"):
    Doc.set_extension("raw_text", default=None)


def is_quote(word: str) -> bool:
    return word[0] in "\"'`"


class Sentence:
    def __init__(self, doc: Doc):
        self.doc = doc

    @cached_property
    def metadata(self):
        return tuple((token.tag_, token.text, token.lemma_) for token in self.doc)

    @cached_property
    def msgpack(self):
        data = BytesIO()

        values = {
            "text": self.doc.text,
            "tokens": self.metadata,
        }

        msgpack.pack(values, data)

        if self.doc.has_vector:
            if not isinstance(self.doc.vector, np.ndarray):
                self.doc._vector = self.doc.vector.get()

            data.seek(0)
            data.write(b"\x83")
            data.seek(0, io.SEEK_END)
            msgpack.pack("vector", data)
            data.write(b"\xc5")
            if sys.byteorder == "big":
                vec = self.doc.vector.newbyteorder("little").tobytes()
            else:
                vec = self.doc.vector.tobytes()
            data.write(len(vec).to_bytes(2, byteorder="big"))
            data.write(vec)

        return data.getbuffer()

    @cached_property
    def json(self):
        values = {
            "text": self.doc.text,
            "tokens": [
                {"tag": t[0], "text": t[1], "lemma": t[2]} for t in self.metadata
            ],
        }

        if self.doc.has_vector:
            if not isinstance(self.doc.vector, np.ndarray):
                self.doc._vector = self.doc.vector.get()
            values["vector"] = [float(x) for x in self.doc.vector]

        return values


class SpacyModel(str, Enum):
    EN_SM = "en_core_web_sm"
    EN_MD = "en_core_web_md"
    EN_LG = "en_core_web_lg"
    EN_TRF = "en_core_web_trf"

    def __str__(self):
        return self.value


class Tokenizer:
    TOKEN_CACHE: Dict[SpacyModel, Dict[str, Sentence]] = defaultdict(dict)
    ENTITY_CACHE = defaultdict(dict)
    TAGGER_CACHE = {}
    CACHE_LOADED = defaultdict(set)

    def __init__(self, model: SpacyModel = SpacyModel.EN_LG):
        self.token_cache = Tokenizer.TOKEN_CACHE[model]
        self.entity_cache = Tokenizer.ENTITY_CACHE[model]
        self.tagger = self.load_tagger(model)

    @classmethod
    def load_tagger(cls, model: SpacyModel):
        if model not in cls.TAGGER_CACHE:
            spacy.prefer_gpu(0)
            LOGGER.info(f"Loading spacy/{model}")
            nlp = spacy.load(str(model))
            nlp.add_pipe("doc_tokens")
            cls.TAGGER_CACHE[model] = nlp
        return cls.TAGGER_CACHE[model]

    @classmethod
    def from_cache(cls, path: Union[Path, str], model: SpacyModel = SpacyModel.EN_LG):
        if path in cls.CACHE_LOADED[model]:
            LOGGER.info(f"Path {path} already cached.")
            return Tokenizer(model)

        tagger = cls.load_tagger(model)
        has_vec = "tok2vec" in tagger.pipe_names
        if has_vec:
            Doc.set_extension("doc_vec", default=None)

        try:
            docs = DocBin(store_user_data=True).from_disk(path).get_docs(tagger.vocab)
            docs = [(doc._.raw_text, doc) for doc in docs]
            if has_vec:
                for (_, doc) in docs:
                    doc._vector = np.array(doc._.doc_vec)

            cls.TOKEN_CACHE[model].update((text, Sentence(doc)) for text, doc in docs)
            cls.CACHE_LOADED[model].add(path)
        except FileNotFoundError:
            pass

        if has_vec:
            Doc.remove_extension("doc_vec")

        return Tokenizer(model)

    def write_data(self, path: Union[Path, str]):
        doc_bin = DocBin(store_user_data=True)

        has_vec = "tok2vec" in self.tagger.pipe_names

        if has_vec:
            Doc.set_extension("doc_vec", default=None)

        for sent in self.token_cache.values():
            if has_vec:
                sent.doc._.doc_vec = sent.doc.vector
            doc_bin.add(sent.doc)

        if isinstance(path, Path):
            path.parent.mkdir(exist_ok=True, parents=True)
        else:
            Path(path).parent.mkdir(exist_ok=True, parents=True)
        doc_bin.to_disk(path)

        if has_vec:
            Doc.remove_extension("doc_vec")

    def tokenize(self, sentence: str, idents=None) -> Sentence:
        """Tokenizes and tags the given sentence."""
        if sentence not in self.token_cache:
            doc = self.tagger(unidecode.unidecode(sentence))
            doc._.raw_text = sentence
            self.token_cache[sentence] = Sentence(doc)

        return self.token_cache[sentence]

    def stream_tokenize(self, sentences: List[str], idents=None) -> Iterable[Sentence]:
        """
        Returns a generator, producing tokenized and tagged sentences in order of their appearance in the input.

        ~2x faster than calling tokenize on an item-by-item basis for 6000 items
        (all unique sentences in stdlib).
        """
        tokenized_sentences = [self.token_cache.get(sentence) for sentence in sentences]
        empty_inds = [i for i, val in enumerate(tokenized_sentences) if val is None]
        new_sents = self.tagger.pipe(
            unidecode.unidecode(sentences[i]) for i in empty_inds
        )

        for i, tokenized in enumerate(tokenized_sentences):
            if tokenized is None:
                tokenized = next(new_sents)
                sent = sentences[i]
                tokenized._.raw_text = sent
                tokenized = Sentence(tokenized)
                self.token_cache[sent] = tokenized
            yield tokenized

    def entities(self, sentence: str) -> dict:
        """Performs NER and SRL analysis of the given sentence, using the models from
        `Combining Formal and Machine Learning Techniques for the Generation of JML Specifications`.
        Output is a dictionary, containing keys "ner" and "srl", corresponding to the NER and SRL entities,
        respectively. The items are formatted as either a dictionary or list of dictionaries for spaCy display."""
        sentence = unidecode.unidecode(sentence).rstrip(".")

        if sentence not in self.entity_cache:
            res = ner_and_srl(sentence)
            ents = []
            for item in res["entities"]:
                ent = {
                    "start": item["pos"],
                    "end": item["pos"] + len(item["text"]),
                    "label": item["type"],
                }
                ents.append(ent)

            spacy_ner = {"text": sentence, "ents": ents}

            spacy_srls = []
            for item in res["predicates"]:
                ents = []
                predicate = item["predicate"]
                predicate.pop("len")
                predicate["start"] = predicate.pop("pos")
                predicate["end"] = len(predicate.pop("text")) + predicate["start"]
                predicate["label"] = "PRED"
                ents.append(predicate)
                for label, metadata in item["roles"].items():
                    ent = {
                        "start": metadata["pos"],
                        "end": metadata["pos"] + len(metadata["text"]),
                        "label": label,
                    }
                    ents.append(ent)
                spacy_srl = {"text": sentence, "ents": ents}
                spacy_srls.append(spacy_srl)

            if not spacy_srls:
                spacy_srls.append({"text": sentence, "ents": []})

            self.entity_cache[sentence] = {"ner": spacy_ner, "srl": spacy_srls}

        return self.entity_cache[sentence]


# confusing examples: log fns, trig fns, pow fns
# TODO: Side effects:
# Assignment operation:
# Assign result of fn to val
# Assign result of operation to val
# eg. Increments a by n
# Decrements a by n
# Divides a by n
# Increases a by n
# Decreases a by n
# Negates a
# Multiplies a by n
# Subtracts n from a
# Adds n to a
# Shifts a to the DIR by n
# DIR shifts a by n
# a is shifted to the right by n
# a is divided by n
# a is multiplied by n
# a is increased by n
# a is incremented by n
# a is decremented by a
# a is negated
# a is right shifted by n
# a is ?VBD?
#         "Returns `true` if and only if `self == 2^k` for some `k`."
#                                                  ^ not as it appears in Rust (programmer error, obviously)
#                                                   (eg. allow mathematical notation?)
#
# TODO: Find examples that are not supported by Prusti
#  async fns (eg. eventually) ? (out of scope)
#  for all / for each
#  Greater than
#  https://doc.rust-lang.org/std/primitive.u32.html#method.checked_next_power_of_two
#  If the next power of two is greater than the type’s maximum value
#  No direct support for existential
#  For any index that meets some condition, x is true. (eg. forall)
