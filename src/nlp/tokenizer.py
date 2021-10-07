import io
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Union
import logging
from io import BytesIO
import sys

import numpy as np
import spacy
from spacy.tokens import Doc, DocBin
import unidecode
import msgpack

try:
    from ner import ner_and_srl
    from fix_tokens import fix_tokens
except ModuleNotFoundError:
    from .ner import ner_and_srl
    from .fix_tokens import fix_tokens

LOGGER = logging.getLogger(__name__)

Doc.set_extension("raw_text", default=None)


def is_quote(word: str) -> bool:
    return word[0] in "\"'`"


class Sentence:
    def __init__(self, doc: Doc):
        self.doc = doc
        self.metadata = tuple((token.tag_, token.text, token.lemma_) for token in self.doc)

    def msgpack(self):
        data = BytesIO()

        values = {
            "text": self.doc.text,
            "tokens": self.metadata,
        }

        msgpack.pack(values, data)

        if self.doc.has_vector:
            data.seek(0)
            data.write(b"\x83")
            data.seek(0, io.SEEK_END)
            msgpack.pack("vector", data)
            data.write(b"\xc5")
            if sys.byteorder == "big":
                vec = self.doc.vector.newbyteorder("little").tobytes()
            else:
                vec = self.doc.vector.tobytes()
            data.write(len(vec).to_bytes(2, byteorder='big'))
            data.write(vec)

        data.seek(0)
        return data.read()


class SpacyModel(str, Enum):
    EN_SM = "en_core_web_sm"
    EN_MD = "en_core_web_md"
    EN_LG = "en_core_web_lg"
    EN_TRF = "en_core_web_trf"

    def __str__(self):
        return self.value


class Tokenizer:
    TOKEN_CACHE = defaultdict(dict)
    ENTITY_CACHE = defaultdict(dict)
    TAGGER_CACHE = {}

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

            cls.TOKEN_CACHE[model].update(docs)
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
            self.token_cache[sentence] = doc

        return Sentence(self.token_cache[sentence])

    def stokenize(self, sentences: List[str], idents=None) -> List[Sentence]:
        """
        Tokenizes and tags the given sentences - 2x faster than tokenize for 6000 items
        (all unique sentences in stdlib).
        """
        sentence_dict = {i: self.token_cache.get(sentence) for i, sentence in enumerate(sentences)}

        empty_inds = [i for i, val in sentence_dict.items() if val is None]
        empty_sents = [unidecode.unidecode(sentences[i]) for i in empty_inds]

        for i, tokenized in zip(empty_inds, self.tagger.pipe(empty_sents)):
            sent = sentences[i]
            tokenized._.raw_text = sent
            self.token_cache[sent] = tokenized
            sentence_dict[i] = tokenized

        return [Sentence(doc) for doc in sentence_dict.values()]

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
                    "label": item["type"]
                }
                ents.append(ent)

            spacy_ner = {
                "text": sentence,
                "ents": ents
            }

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
                        "label": label
                    }
                    ents.append(ent)
                spacy_srl = {
                    "text": sentence,
                    "ents": ents
                }
                spacy_srls.append(spacy_srl)

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
