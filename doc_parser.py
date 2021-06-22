from enum import Enum, auto
from typing import List, Union

import flair.data
from flair.data import Sentence, Label, Token
from flair.models import MultiTagger

from nltk.tree import Tree
import nltk


class CodeTokenizer(flair.data.Tokenizer):
    def tokenize(self, sentence: str) -> List[Token]:
        parts = []
        num = 1
        a = 0
        while sentence[a].isspace() and a < len(sentence):
            a += 1

        while a < len(sentence):
            if sentence[a] == "`":
                ind = sentence[a + 1:].find("`")
                if ind != -1:
                    ind += 2
            else:
                ind = sentence[a:].find(" ")
            if ind == -1:
                t = Token(sentence[a:], num, whitespace_after=False, start_position=a)
                parts.append(t)
                break
            else:
                t = Token(sentence[a: a + ind], num, whitespace_after=True, start_position=a)
                parts.append(t)

            num += 1
            a = a + ind
            while a < len(sentence) and sentence[a] == " ":
                a += 1

        return parts


def print_tokens(token_dict):
    for entity in token_dict["entities"]:
        print(entity["text"], entity["labels"][0])
    print()


def might_be_code(s: str):
    # detect bool literals
    if s.lower() in {"true", "false", "self"}:
        return True

    # detect numbers
    if s.endswith(("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64")):
        return True

    if s.isnumeric():
        return True

    return False


def correct_tokens(token_dict):
    entities = token_dict["entities"]

    for entity in entities:
        if entity["text"].startswith("`"):
            entity["labels"][0] = Label("CODE")

    for i, entity in enumerate(entities):
        if might_be_code(entity["text"]):
            entity["labels"][0] = Label("LIT")

    if len(entities) <= 2:
        return

    for i, entity in enumerate(entities):
        if i == 0 and entity["text"].lower() == "returns":
            if i + 1 < len(entities) and entities[i + 1]["labels"][0].value in {
                "NN", "NNP", "NNPS", "NNS", "CODE", "LIT"
            }:
                entity["labels"][0] = Label("RET")


class Code:
    def __init__(self, tree: Tree):
        self.code: str = tree[0]

    def __str__(self):
        return self.code.strip("`")


class Literal:
    def __init__(self, tree: Tree):
        self.str: str = tree[0]

    def __str__(self):
        return self.str


class Object:
    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in {("PRP",), ("DT", "NN"), ("CODE",), ("LIT",)}:
            raise ValueError(f"Bad tree - expected OBJ productions, got {labels}")
        self.labels = labels
        self.tree = tree

    def as_code(self):
        if self.labels == ("CODE",):
            return Code(self.tree[0])
        if self.labels == ("LIT",):
            return Literal(self.tree[0])
        if self.labels == ("DT", "NN"):
            # TODO: This assumes that val in "The val" is a variable.
            return self.tree[1][0]
        raise ValueError(f"{self.labels} not handled.")


class Comparator(Enum):
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    EQ = auto()
    NEQ = auto()

    @classmethod
    def from_str(cls, s: str):
        return {
            "less": cls.LT,
            "greater": cls.GT,
            "equal": cls.EQ,
        }[s]

    def __str__(self):
        return {
            self.LT: "<",
            self.GT: ">",
            self.LTE: "<=",
            self.GTE: ">=",
            self.EQ: "==",
            self.NEQ: "!=",
        }[self]

    def apply_eq(self):
        return {
            self.LT: self.LTE,
            self.GT: self.GTE,
        }.get(self, self)

    def negate(self, negate):
        if not negate:
            return self

        return {
            self.LT: self.GTE,
            self.GT: self.LTE,
            self.LTE: self.GT,
            self.GTE: self.LT,
            self.EQ: self.NEQ,
            self.NEQ: self.EQ,
        }[self]


class Relation:
    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in {("JJR", "IN", "OBJ"), ("JJR", "EQTO", "OBJ"), ("JJ", "IN", "OBJ")}:
            raise ValueError(f"Bad tree - expected REL productions, got {labels}")
        self.labels = labels
        self.tree = tree
        self.negate = False

    def as_code(self):
        if self.labels in {("JJR", "IN", "OBJ"), ("JJ", "IN", "OBJ")}:
            relation = Comparator.from_str(self.tree[0][0]).negate(self.negate)
            return f" {relation} {Object(self.tree[2]).as_code()}"
        if self.labels == ("JJR", "EQTO", "OBJ"):
            relation = Comparator.from_str(self.tree[0][0]).apply_eq().negate(self.negate)
            return f" {relation} {Object(self.tree[2]).as_code()}"

        raise ValueError(f"Case {self.labels} not handled.")

    def apply_adverb(self, s: str):
        if s.lower() != "not":
            raise ValueError(f"Only not is supported as an adverb for Relation: got {s}")
        self.negate = True
        return self


class Property:
    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in {("VBP", "JJ"), ("VBZ", "JJ"), ("VBZ", "REL"), ("VBZ", "RB", "REL"), ("VBZ", "LIT")}:
            raise ValueError(f"Bad tree - expected PROP productions, got {labels}")
        self.labels = labels
        self.tree = tree

    def as_code(self):
        if self.labels == ("VBZ", "JJ"):
            return f".{self.tree[1][0]}()"
        if self.labels == ("VBZ", "LIT"):
            return f" == {self.tree[1][0]}"
        if self.labels == ("VBZ", "REL"):
            return Relation(self.tree[1]).as_code()
        if self.labels == ("VBZ", "RB", "REL"):
            return Relation(self.tree[2]).apply_adverb(self.tree[1][0]).as_code()

        raise ValueError(f"Case {self.labels} not handled.")


class Assert:
    def __init__(self, tree: Tree):
        if tree[0].label() != "OBJ":
            raise ValueError(f"Bad tree - expected OBJ, got {tree[0].label()}")
        if tree[1].label() != "PROP":
            raise ValueError(f"Bad tree - expected PROP, got {tree[1].label()}")

        self.obj = Object(tree[0])
        self.prop = Property(tree[1])

    def as_code(self):
        return f"{self.obj.as_code()}{self.prop.as_code()}"


class Predicate:
    def __init__(self, tree: Tree):
        if tree[0].label() not in {"IFF", "IN"}:
            raise ValueError(f"Bad tree - expected IFF or IN, got {tree[0].label()}")
        if tree[1].label() != "ASSERT":
            raise ValueError(f"Bad tree - expected ASSERT, got {tree[1].label()}")

        self.iff = tree[0].label() == "IFF"
        self.assertion = Assert(tree[1])

    def as_code(self):
        return self.assertion.as_code()


class ReturnIf:
    def __init__(self, tree: Union[Tree, List[Tree]]):
        if tree[0].label() != "RET":
            raise ValueError(f"Bad tree - expected RET, got {tree[0].label()}")

        if tree[1].label() != "OBJ":
            raise ValueError(f"Bad tree - expected OBJ, got {tree[1].label()}")

        if tree[2].label() not in {"EPRED", "PRED"}:
            raise ValueError(f"Bad tree - expected EPRED or PRED, got {tree[2].label()}")

        # Need to find out what the ret value is.

        self.ret_val = Object(tree[1])
        self.pred = Predicate(tree[2])

    def __str__(self):
        return f"return {self.ret_val.as_code()};"

    def as_spec(self):
        if self.pred.iff:
            return f"""#[ensures({self.pred.as_code()} ==> (result == {self.ret_val.as_code()}))]
#[ensures((result == {self.ret_val.as_code()}) ==> {self.pred.as_code()})]"""
        return f"""#[ensures({self.pred.as_code()} ==> (result == {self.ret_val.as_code()}))]"""


class IfReturn(ReturnIf):
    def __init__(self, tree: Tree):
        super().__init__([subtree for subtree in tree[::-1]])


class Existential:
    def __init__(self, tree: Tree):
        pass


class Specification:
    def __init__(self, tree: Tree):
        self.spec = {
            "RETIF": ReturnIf,
            "IFRET": IfReturn,
        }[tree[0].label()](tree[0])

    def as_spec(self):
        return self.spec.as_spec()


def get_bottom_nodes(tree: Tree, nodes: List[Tree]):
    if isinstance(tree, str):
        return nodes

    if len(tree) == 1 and isinstance(tree[0], str):
        nodes.append(tree)
    else:
        for sub_tree in tree:
            get_bottom_nodes(sub_tree, nodes)

    return nodes


def attach_words_to_nodes(tree: Tree, words: List[str]):
    nodes = get_bottom_nodes(tree, [])
    for node, word in zip(nodes, words):
        node[0] = word

    return tree


class POSModel(Enum):
    POS = auto()
    POS_FAST = auto()
    UPOS = auto()
    UPOS_FAST = auto()

    def __str__(self):
        return {
            POSModel.POS: "pos",
            POSModel.POS_FAST: "pos-fast",
            POSModel.UPOS: "upos",
            POSModel.UPOS_FAST: "upos-fast"
        }[self]


class Parser:
    def __init__(self, pos_model: POSModel = POSModel.POS, grammar_path: str = "codegrammar.cfg"):
        self.root_tagger = MultiTagger.load([str(pos_model)])
        self.tokenizer = CodeTokenizer()

        self.grammar = nltk.data.load(f"file:{grammar_path}")
        self.rd_parser = nltk.RecursiveDescentParser(self.grammar)

    def parse_sentence(self, sentence: str):
        sentence = Sentence(sentence, use_tokenizer=self.tokenizer)
        self.root_tagger.predict(sentence)
        token_dict = sentence.to_dict(tag_type="pos")
        correct_tokens(token_dict)
        labels = [entity["labels"][0] for entity in token_dict["entities"]]
        words = [entity["text"] for entity in token_dict["entities"]]
        print(labels)
        nltk_sent = [label.value for label in labels]
        return [
            attach_words_to_nodes(tree, words)
            for tree in self.rd_parser.parse(nltk_sent)
        ]


def main():
    predicates = [
        "I am red",
        "It is red",
        "The index is less than `self.len()`",
        "The index is 0u32",
        # Well handled.
        "Returns `true` if and only if `self` is 0u32",
        "Returns `true` if `self` is 0u32",
        "Returns true if self is 0u32",
        # Need to find a corresponding function or method for this case.
        "Returns `true` if and only if `self` is blue",
        "Returns `true` if the index is less than `self.len()`",
        "Returns `true` if the index is greater than or equal to `self.len()`",
        "Returns `true` if the index is equal to `self.len()`",
        "Returns `true` if the index is not equal to `self.len()`",
    ]

    parser = Parser()
    for sentence in predicates:
        print("=" * 80)
        print("Sentence:", sentence)
        print("=" * 80)
        for tree in parser.parse_sentence(sentence):
            tree: nltk.tree.Tree = tree
            try:
                spec = Specification(tree)
                print(spec.as_spec())
            except LookupError as e:
                print(f"No specification found. {e}")
            print(tree)
            print()


if __name__ == '__main__':
    main()
