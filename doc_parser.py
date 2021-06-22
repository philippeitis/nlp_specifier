from enum import Enum, auto
from typing import List

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


def correct_tokens(token_dict):
    entities = token_dict["entities"]

    for entity in entities:
        if entity["text"].startswith("`"):
            entity["labels"][0] = Label("CODE")

    if len(entities) <= 2:
        return

    for i, entity in enumerate(entities):
        if i == 0 and entity["text"].lower() == "returns":
            if i + 1 < len(entities) and entities[i + 1]["labels"][0].value in {"NN", "NNP", "NNPS", "NNS", "CODE"}:
                entity["labels"][0] = Label("RET")


class Code:
    def __init__(self, tree: Tree):
        self.code: str = tree[0]

    def __str__(self):
        return self.code.strip("`")


class Object:
    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in {("PRP",), ("DT", "NN"), ("CODE",)}:
            raise ValueError(f"Bad tree - expected OBJ productions, got {labels}")
        self.labels = labels
        self.tree = tree

    def as_code(self):
        if self.labels == ("CODE",):
            return Code(self.tree[0])


class Property:
    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in {("VBP", "JJ"), ("VBZ", "JJ"), ("VBZ", "REL")}:
            raise ValueError(f"Bad tree - expected PROP productions, got {labels}")
        self.labels = labels
        self.tree = tree

    def as_code(self):
        if self.labels == ("VBZ", "JJ"):
            return f"{self.tree[1][0]}"


class Assert:
    def __init__(self, tree: Tree):
        if tree[0].label() != "OBJ":
            raise ValueError(f"Bad tree - expected OBJ, got {tree[0].label()}")
        if tree[1].label() != "PROP":
            raise ValueError(f"Bad tree - expected PROP, got {tree[1].label()}")

        self.obj = Object(tree[0])
        self.prop = Property(tree[1])

    def as_code(self):
        return f"{self.obj.as_code()}.{self.prop.as_code()}()"


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
    def __init__(self, tree: Tree):
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
            return f"""#[ensures((result == {self.ret_val.as_code()}) ==> {self.pred.as_code()})]
#[ensures({self.pred.as_code()} ==> (result == {self.ret_val.as_code()}))]"""


class Existential:
    def __init__(self, tree: Tree):
        pass


class Specification:
    def __init__(self, tree: Tree):
        pass


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
        "The index is small",
        "Returns `true` if and only if `self` is blue"
    ]

    parser = Parser()
    for sentence in predicates:
        print("=" * 80)
        print("Sentence:", sentence)
        print("=" * 80)
        for tree in parser.parse_sentence(sentence):
            tree: nltk.tree.Tree = tree
            if tree[0].label() == "RETIF":
                print(ReturnIf(tree[0]).as_spec())
            print(tree)
            print()


if __name__ == '__main__':
    main()
