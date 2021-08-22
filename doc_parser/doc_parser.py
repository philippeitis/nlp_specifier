from collections import defaultdict
from copy import copy
from typing import List, Iterator, Tuple
import logging
import os

import nltk
from nltk.tree import Tree
import spacy

from fix_tokens import fix_tokens

GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "codegrammar.cfg")
logger = logging.getLogger("flair")
logger.setLevel(logging.ERROR)


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


class Parser:
    TREE_CACHE = defaultdict(dict)
    TOKEN_CACHE = defaultdict(dict)
    TAGGER_CACHE = {}

    def __init__(self, grammar: str, model: str = "en_core_web_sm"):
        self.tree_cache = Parser.TREE_CACHE[model]
        self.token_cache = Parser.TOKEN_CACHE[model]
        if model not in Parser.TAGGER_CACHE:
            Parser.TAGGER_CACHE[model] = spacy.load(model)
        self.tagger = Parser.TAGGER_CACHE[model]

        self.grammar = nltk.CFG.fromstring(grammar)
        self.rd_parser = nltk.ChartParser(self.grammar)

    @classmethod
    def from_path(cls, grammar_path: str, model: str = "en_core_web_sm"):
        with open(grammar_path) as f:
            return cls(f.read(), model)

    @classmethod
    def default(cls):
        return cls.from_path(grammar_path=GRAMMAR_PATH)

    def tokenize_sentence(self, sentence: str, idents=None) -> Tuple[List[str], List[str]]:
        sentence = sentence \
            .replace("isn't", "is not") \
            .rstrip(".")

        if sentence not in self.token_cache:
            doc = self.tagger(sentence)
            self.token_cache[sentence] = doc

        doc = copy(self.token_cache[sentence])

        fix_tokens(doc, idents=idents)
        labels: List[str] = [token.tag_ for token in doc]
        words = [token.text for token in doc]
        return labels, words

    def parse_sentence(self, sentence: str, idents=None, attach_tags=True) -> Tree:
        """Parses the sentence, using `idents` to detect values in the text.
        If `idents` is not provided, no IDENT literals will appear in the tree.
        If `attach_tags` is set, labels will transformed for usage in an augmented CFG.
        Otherwise, labels will remain unmodified.
        """
        labels, words = self.tokenize_sentence(sentence, idents)
        if attach_tags:
            nltk_sent = [label if is_quote(word) or word in ".,\"'`" else f"{word}_{label}"
                         for label, word in zip(labels, words)]
        else:
            nltk_sent = [label for label in labels]

        nltk_str = " ".join(nltk_sent)
        if nltk_str not in self.tree_cache:
            self.tree_cache[nltk_str] = ParseStorage(self.rd_parser.parse(nltk_sent))

        for tree in self.tree_cache[nltk_str]:
            yield replace_leaf_nodes(tree, iter(words))


def main():
    test_cases = [
        ("Returns `true` if `self` is 0u32", "#[ensures(self == 0u32 ==> result)]")
    ]
    predicates = [
        # "I am red",
        # "It is red",
        # "The index is less than `self.len()`",
        # "The index is 0u32",
        # Well handled.
        # "Returns `true` if and only if `self` is 0u32",
        # "Returns `true` if `self` is 0u32",
        # "Returns true if self is 0u32",
        # # Need to find a corresponding function or method for this case.
        # "Returns `true` if and only if `self` is blue",
        # # Well handled.
        # "Returns `true` if the index is greater than or equal to `self.len()`",
        # "Returns `true` if the index is equal to `self.len()`",
        # "Returns `true` if the index is not equal to `self.len()`",
        # "Returns `true` if the index isn't equal to `self.len()`.",
        # "Returns `true` if the index is not less than `self.len()`",
        # "Returns `true` if the index is smaller than or equal to `self.len()`",
        # "If the index is smaller than or equal to `self.len()`, returns true.",
        # "`index` must be less than `self.len()`",
        # "for each index `i` from `0` to `self.len()`, `i` must be less than `self.len()`",
        # "for each index `i` up to `self.len()`, `i` must be less than `self.len()`",
        # "for each index `i` from `0` to `self.len()` inclusive, `i` must be less than `self.len()`",
        # "`i` must be less than `self.len()`",
        # "for each index `i` from `0` to `self.len()` inclusive, `i` must be less than or equal to `self.len()`",
        # "for each index `i` from `0` to `self.len()`, `self.lookup(i)` must not be equal to 0",
        # # differentiate btw will and must wrt result
        # "for each index `i` from `0` to `self.len()`, `self.lookup(i)` will not be the same as `result.lookup(i)`",
        # "For all indices `i` from 0 to 32, `result.lookup(i)` will be equal to `self.lookup(31 - i)`",
        # "For all indices `i` between 0 and 32, and less than `amt`, `result.lookup(i)` will be false.",
        # # "For all indices `i` between 0 and 32, or less than `amt`, `result.lookup(i)` will be false.",
        # "`self.lookup(index)` will be equal to `val`",
        # "For all `i` between 0 and 32, and not equal to `index`, `self.lookup(i)` will be unchanged.",
        # "For all `i` between 0 and 32, and not equal to `index`, `self.lookup(i)` will not change.",
        # "Returns true if self is not blue",
        # "`self` must be blue",
        # "`other.v` must not be equal to 0",
        # "For all `i` between 0 and 32, and not equal to `index`, `self.lookup(i)` will remain static.",
        # "Returns `true` if and only if `self == 2^k` for all `k`.",
        # "Returns `true` if and only if `self == 2^k` for any `k`."
        # "For each index from 0 to 5, `self.lookup(index)` must be true.",
        # "For each index up to 5, `self.lookup(index)` must be true.",
        # "`a` must be between 0 and `self.len()`.",
        # "`a` and `b` must be equal to 0 or `self.len()`.",
        # "True is returned.",
        # "Returns `a` if `self.check(b)`, otherwise `b`",
        # "Returns `a` if `fn(a)`",
        # "If `fn(a)`, returns `a`",
        # "If `fn(a)`, returns `a`, otherwise `b`",
        # "Returns `a` logical and `b`",
        # "Returns `a` bitwise and `b`",
        # "Returns `a` xor `b`",
        # "Returns `a` multiplied by `b`",
        # "Returns `a` divided by `b`",
        # "Returns `a` subtracted by `b`",
        # "Returns `a` subtracted from `b`",
        # "Returns `a` added to `b`",
        # "Returns `a` shifted to the right by `b`",
        # "Returns `a` shifted to the left by `b`",
        # "Returns `a` left shift `b`",
        # "Returns `a` right shift `b`",
        # TODO: Side effects:
        #  Assignment operation:
        #  Assign result of fn to val
        #  Assign result of operation to val
        #   eg. Increments a by n
        #       Decrements a by n
        #       Divides a by n
        #       Increases a by n
        #       Decreases a by n
        #       Negates a
        #       Multiplies a by n
        #       Subtracts n from a
        #       Adds n to a
        #       Shifts a to the DIR by n
        #       DIR shifts a by n
        #       a is shifted to the right by n
        #       a is divided by n
        #       a is multiplied by n
        #       a is increased by n
        #       a is incremented by n
        #       a is decremented by a
        #       a is negated
        #       a is right shifted by n
        #       a is ?VB?
        # "Sets `a` to 1",
        # "Assigns 1 to `a`.",
        # "Increments `a` by 1",
        # "Adds 1 to `a`"
        "`a` is incremented by 1",
        "`a` is negated",
        "`a` is right shifted by `n`",
    ]
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
    parser = Parser.default()
    for sentence in predicates:
        print("=" * 80)
        print("Sentence:", sentence)
        print("=" * 80)
        for tree in parser.parse_sentence(sentence, attach_tags=False):
            # tree: Tree = tree
            # try:
            #     print(Specification(tree).as_spec())
            # except LookupError as e:
            #     print(f"No specification found: {e}")
            # except UnsupportedSpec as s:
            #     print(f"Specification element not supported ({s})")
            print(tree)
            print()

    # for sentence, expected in test_cases:
    #     try:
    #         tree = next(parser.parse_sentence(sentence))
    #         assert Specification(tree).as_spec() == expected
    #     except AssertionError as a:
    #         print(a)


if __name__ == "__main__":
    main()
# confusing examples: log fns, trig fns, pow fns
