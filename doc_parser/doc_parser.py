from enum import Enum
from typing import List, Tuple, Iterator, Iterable, Optional
import logging
import os

from flair.data import Sentence, Label
from flair.models import MultiTagger
import nltk
from nltk.tree import Tree

from lemmatizer import lemmatize
from tokenizer import CodeTokenizer

GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "codegrammar.cfg")
logger = logging.getLogger("flair")
logger.setLevel(logging.ERROR)


def is_quote(word: str) -> bool:
    return word[0] in "\"'`"


def get_literal_tag(word: str, idents=None) -> Tuple[Optional[str], str]:
    """Determines whether `word` matches the pattern for a code literal - specifically,
     if the word is a Rust literal or a function argument.

    :param word:
    :param idents:
    :return:
    """
    # detect bool literals
    if word.lower() in {"true", "false"}:
        return "LIT_BOOL", word.lower()

    if idents and word in idents:
        return "IDENT", word

    # detect numbers
    if word.endswith(("u8", "u16", "u32", "u64", "usize")):
        return "LIT_UNUM", word

    if word.endswith(("i8", "i16", "i32", "i64", "isize")):
        return "LIT_INUM", word

    if word.endswith(("f32", "f64")):
        return "LIT_FNUM", word

    if word.isnumeric():
        return "LIT_FNUM" if "." in word else "LIT_INT", word

    if word[0] == "\"":
        return "LIT_STR", word

    if word[0] == "'":
        return "LIT_CHAR", word

    return None, word


def apply_operation_tokens(token_dict):
    entities = token_dict["entities"]

    def get_label(t) -> str:
        return t["labels"][0].value

    def set_label(t, label: str):
        t["labels"][0] = Label(label)

    def text(t) -> str:
        return t["text"]

    def is_obj(t: str) -> bool:
        return get_label(t) in {
            "NN", "NNP", "NNPS", "NNS", "CODE", "LIT"
        }

    def is_arith(t) -> bool:
        # short-hand / passive
        if text(t).lower() in {
            "add", "plus", "subtract", "sub", "divide", "div", "multiply", "mul", "remainder",
            "rem"
        }:
            return True
        # past-tense
        return text(t).lower() in {"added", "subtracted", "divided", "multiplied"}

    for i, entity in enumerate(entities):
        if i == 0:
            continue

        if is_obj(entity):
            delta = len(entities) - i - 1
            if delta >= 6 and is_obj(entities[i + 6]):
                # SHIFT IN DT NN IN
                if lemmatize(text(entities[i + 1]).lower(), "v") == "shift" \
                        and get_label(entities[i + 2]) == "IN" \
                        and get_label(entities[i + 3]) == "DT" \
                        and get_label(entities[i + 4]).startswith("NN") \
                        and get_label(entities[i + 5]) == "IN":
                    set_label(entities[i + 1], "SHIFT")
            if delta >= 3 and is_obj(entities[i + 3]):
                # JJ SHIFT
                if get_label(entities[i + 1]).startswith("JJ") \
                        and lemmatize(text(entities[i + 2]).lower(), "v") == "shift":
                    set_label(entities[i + 2], "SHIFT")
                # ARITH IN
                elif is_arith(entities[i + 1]) \
                        and get_label(entities[i + 2]) == "IN":
                    set_label(entities[i + 1], "ARITH")
            # ARITH
            if delta >= 2 and is_obj(entities[i + 2]):
                if is_arith(entities[i + 1]) or text(entities[i + 1]).lower() == "xor":
                    set_label(entities[i + 1], "ARITH")


def fix_tokens(token_dict, idents=None):
    def get_label(t) -> str:
        return t["labels"][0].value

    def set_label(t, label: Label):
        t["labels"][0] = label

    def text(t) -> str:
        return t["text"]

    def set_text(t, word: str):
        t["text"] = word

    def is_obj(t: str) -> bool:
        return t in {
            "NN", "NNP", "NNPS", "NNS", "CODE", "LIT"
        }

    entities = token_dict["entities"]

    for entity in entities:
        if text(entity).startswith("`"):
            entity["labels"][0] = Label("CODE")

    for i, entity in enumerate(entities):
        lit_tag, word = get_literal_tag(text(entity), idents)
        if lit_tag:
            set_label(entity, Label("LIT"))
            set_text(entity, word)

    for i, entity in enumerate(entities):
        try:
            word = lemmatize(text(entity).lower(), get_label(entity))
        except ValueError:
            continue
        if word == "return":
            if get_label(entity).startswith("VB"):
                set_label(entity, Label("RET"))
            elif i + 1 < len(entities) and is_obj(get_label(entities[i + 1])):
                set_label(entity, Label("RET"))
            elif i + 2 < len(entities) and is_obj(get_label(entities[i + 2])):
                set_label(entity, Label("RET"))
            elif i >= 2 and is_obj(get_label(entities[i - 2])) and get_label(entities[i - 1]) in {"VBZ"}:
                set_label(entity, Label("RET"))
            elif i >= 1 and is_obj(get_label(entities[i - 1])):
                set_label(entity, Label("RET"))
        elif word == "if":
            set_label(entity, Label("IF"))

    if len(entities) <= 2:
        return

    for i, entity in enumerate(entities):
        if text(entity).lower() == "for":
            if i + 1 < len(entities) and get_label(entities[i + 1]) in {"DT"}:
                set_label(entity, Label("FOR"))

    apply_operation_tokens(token_dict)


def replace_leaf_nodes(tree: Tree, values: Iterator[str]):
    """Replaces leaf nodes in `tree` with words from `values`.

    :param tree:
    :param values:
    :return:
    """
    if isinstance(tree, str):
        return

    if len(tree) == 1 and isinstance(tree[0], str):
        tree[0] = next(values)
    else:
        for sub_tree in tree:
            replace_leaf_nodes(sub_tree, values)


def attach_words_to_nodes(tree: Tree, words: Iterable[str]) -> Tree:
    """Modifies `tree` in place, by assigning each word in words to a leaf node in tree, in sequential order.

    :param tree:
    :param words:
    :return: modified tree
    """
    replace_leaf_nodes(tree, iter(words))
    return tree


class POSModel(str, Enum):
    POS = "pos"
    POS_FAST = "pos-fast"
    UPOS = "upos"
    UPOS_FAST = "upos-fast"

    def __str__(self):
        return self.value


class Parser:
    def __init__(self, grammar: str, pos_model: POSModel = POSModel.POS):
        self.root_tagger = MultiTagger.load([str(pos_model)])
        self.tokenizer = CodeTokenizer()

        self.grammar = nltk.CFG.fromstring(grammar)
        self.rd_parser = nltk.ChartParser(self.grammar)

    @classmethod
    def from_path(cls, grammar_path: str, pos_model: POSModel = POSModel.POS):
        with open(grammar_path) as f:
            return cls(f.read(), pos_model)

    @classmethod
    def default(cls):
        return cls.from_path(grammar_path=GRAMMAR_PATH)

    def tokenize_sentence(self, sentence: str, idents=None):
        sentence = sentence \
            .replace("isn't", "is not") \
            .rstrip(".")

        sentence = Sentence(sentence, use_tokenizer=self.tokenizer)
        self.root_tagger.predict(sentence)

        token_dict = sentence.to_dict(tag_type="pos")
        fix_tokens(token_dict, idents=idents)
        labels: List[Label] = [entity["labels"][0] for entity in token_dict["entities"]]
        words = [entity["text"] for entity in token_dict["entities"]]
        return labels, words

    def parse_sentence(self, sentence: str, idents=None, attach_tags=True) -> Tree:
        """Parses the sentence, using `idents` to detect values in the text.
        If `idents` is not provided, no IDENT literals will appear in the tree.
        If `attach_tags` is set, labels will transformed for usage in an augmented CFG.
        Otherwise, labels will remain unmodified.
        """
        labels, words = self.tokenize_sentence(sentence, idents)
        if attach_tags:
            nltk_sent = [label.value if is_quote(word) else f"{word}_{label.value}"
                         for label, word in zip(labels, words)]
        else:
            nltk_sent = [label.value for label in labels]

        for tree in self.rd_parser.parse(nltk_sent):
            yield attach_words_to_nodes(tree, words)


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
