from collections import defaultdict
from copy import copy
from enum import Enum
from typing import Iterator, Set, Union
import logging
from pathlib import Path

import nltk
from nltk.tree import Tree
import spacy
from spacy.tokens import Doc
import unidecode

from ner import ner_and_srl
from fix_tokens import fix_tokens

GRAMMAR_PATH = Path(__file__).parent / Path("codegrammar.cfg")
LOGGER = logging.getLogger(__name__)


def is_quote(word: str) -> bool:
    return word[0] in "\"'`"


def replace_leaf_nodes(tree: Tree, values: Iterator[str]):
    """Modifies `tree` in place, by assigning each word in `words` to the leaf nodes in `tree``,
    in sequential order from left to right.

    :param tree: nltk.Tree
    :param values: An iterator of strings.
    :return: modified tree
    """
    if isinstance(tree, str):
        return tree

    if len(tree) == 1 and isinstance(tree[0], str):
        tree[0] = next(values)
    else:
        for sub_tree in tree:
            replace_leaf_nodes(sub_tree, values)
    return tree


class ParseStorage:
    def __init__(self, parse_iter):
        self.parse_iter = parse_iter
        self.items = []

    def add_item(self) -> bool:
        try:
            self.items.append(next(self.parse_iter))
            return True
        except StopIteration:
            return False

    def __iter__(self):
        return ParseIter(self)


class ParseIter:
    def __init__(self, parse_storage):
        self.index = 0
        self.parse_storage = parse_storage

    def __next__(self):
        if self.index < len(self.parse_storage.items):
            index = self.index
            self.index += 1
            return self.parse_storage.items[index]
        if self.parse_storage.add_item():
            index = self.index
            self.index += 1
            return self.parse_storage.items[index]
        raise StopIteration()


class Sentence:
    def __init__(self, doc: Doc):
        self.doc = doc
        self.tags = tuple(token.tag_ for token in self.doc)
        self.words = tuple(token.text for token in self.doc)


class SpacyModel(str, Enum):
    EN_SM = "en_core_web_sm"
    EN_MD = "en_core_web_md"
    EN_LG = "en_core_web_lg"
    EN_TRF = "en_core_web_trf"

    def __str__(self):
        return self.value


class Parser:
    TREE_CACHE = defaultdict(dict)
    TOKEN_CACHE = defaultdict(dict)
    ENTITY_CACHE = defaultdict(dict)
    TAGGER_CACHE = {}

    def __init__(self, grammar: str, model: SpacyModel = SpacyModel.EN_SM):
        if model not in Parser.TAGGER_CACHE:
            LOGGER.info(f"Loading spacy/{model}")
            Parser.TAGGER_CACHE[model] = spacy.load(str(model))

        self.token_cache = Parser.TOKEN_CACHE[model]
        self.entity_cache = Parser.ENTITY_CACHE[model]
        self.tree_cache = Parser.TREE_CACHE[model]
        self.tagger = Parser.TAGGER_CACHE[model]
        self.grammar = nltk.CFG.fromstring(grammar)
        self.tree_parser = nltk.ChartParser(self.grammar)

    @classmethod
    def from_path(cls, grammar_path: Union[str, Path], model: SpacyModel = SpacyModel.EN_SM):
        with open(grammar_path) as f:
            return cls(f.read(), model)

    @classmethod
    def default(cls):
        return cls.from_path(grammar_path=GRAMMAR_PATH)

    def tokens(self) -> Set[str]:
        """Returns the tokens that might appear in the output of parse_tree"""
        return {str(p._lhs) for p in self.grammar._productions}

    def tokenize(self, sentence: str, idents=None) -> Sentence:
        """Tokenizes and tags the given sentence."""
        sentence = unidecode.unidecode(sentence).rstrip(".")

        if sentence not in self.token_cache:
            doc = self.tagger(sentence)
            self.token_cache[sentence] = doc

        doc = copy(self.token_cache[sentence])
        fix_tokens(doc, idents=idents)

        return Sentence(doc)

    def parse_tree(self, sentence: str, idents=None, attach_tags=True) -> Tree:
        """Parses the sentence, using `idents` to detect values in the text.
        If `idents` is not provided, no IDENT literals will appear in the tree.
        If `attach_tags` is set, labels will transformed for usage in an augmented CFG.
        Otherwise, labels will remain unmodified.
        """
        sent = self.tokenize(sentence, idents)
        if attach_tags:
            nltk_sent = [label if is_quote(word) or word in ".,\"'`!()[]{}-" else f"{word}_{label}"
                         for label, word in zip(sent.tags, sent.words)]
        else:
            nltk_sent = sent.tags

        nltk_str = " ".join(nltk_sent)
        if nltk_str not in self.tree_cache:
            self.tree_cache[nltk_str] = ParseStorage(self.tree_parser.parse(nltk_sent))

        for tree in self.tree_cache[nltk_str]:
            yield replace_leaf_nodes(tree, iter(sent.words))

    def entities(self, sentence: str) -> dict:
        """Performs NER and SRL analysis of the given sentence, using the models from
        `Combining Formal and Machine Learning Techniques for the Generation of JML Specifications`.
        Output is a dictionary, containing keys "ner" and "srl", corresponding to the NER and SRL entities,
        respectively. The items are formatted as either a dictionary or list of dictionaries, formatted in the"""
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
