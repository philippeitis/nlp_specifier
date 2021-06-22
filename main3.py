from copy import copy
from typing import List

import flair.data
from flair.data import Sentence, Label, Token
from flair.models import MultiTagger

import nltk

class CodeTokenizer(flair.data.Tokenizer):
    def tokenize(self, sentence: str) -> List[Token]:
        def read_to_next_space(s: str) -> int:
            return s.find(" ")

        def read_to_end_of_code_block(s: str) -> int:
            return s.find("`")

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
                ind = read_to_next_space(sentence[a:])
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


class Production:
    def __init__(self, root, idx, pieces: [str]):
        self.root = root
        self.idx = idx
        self.pieces = pieces

    def all_matches(self, labels: [Label]):
        matches = []
        # Check if start, and check if remaining start w/ curr value and so on
        for j, label in enumerate(labels[:len(labels) - len(self.pieces) + 1]):
            if label.value == self.pieces[0]:
                num = 1
                for labelx, piece in zip(labels[j + 1:], self.pieces[1:]):
                    if labelx.value == piece:
                        num += 1
                    else:
                        break
                if num == len(self.pieces):
                    matches.append((j, j + num))
        return matches


class ProductionRule:
    def __init__(self, name, productions: [List[str]]):
        self.name = name
        self.productions = [Production(name, i, prod) for i, prod in enumerate(productions)]

    def all_matches(self, labels: [Label]):
        matches = {}
        for production in self.productions:
            prod_matches = production.all_matches(labels)
            if prod_matches:
                matches[f"{self.name}-{production.idx}"] = prod_matches
        return matches


class TreeLabel(Label):
    def __init__(self, value, score=1., children=None):
        super().__init__(value, score)
        self.children = children

    def __repr__(self):
        if self.children:
            children = ", ".join([str(child) for child in self.children])
            return f"({self.value} ({self._score}) ({children}))"
        return f"{self.value} ({self._score})"


def replace_fragments(sentence: List[Label], name, matches):
    sentence = copy(sentence)
    for start, end in reversed(matches):
        label = TreeLabel(name, children=sentence[start: end])
        del sentence[start: end]
        sentence.insert(start, label)
    return sentence


def extract_all_trees(prod_rules, labels: [Label]):
    found_match = True
    while found_match:
        found_match = False
        for prod_rule in prod_rules:
            matches = prod_rule.all_matches(labels)
            if matches:
                found_match = True
                for sub_matches in matches.values():
                    labels = replace_fragments(labels, prod_rule.name, sub_matches)
    return labels


class Assert:
    pass


if __name__ == '__main__':
    # load tagger for POS and NER
    tagger = MultiTagger.load(['pos'])

    # make example sentence
    # sentence = Sentence("If I am red, then true is returned.")
    #
    # res = tagger.predict(sentence)
    #
    # print(sentence.tokens)
    # sentence_dict = sentence.to_dict(tag_type="pos")
    # for word in sentence_dict["entities"]:
    #     if word["text"] in {"If", "if", "IF", "iF"}:
    #         print(type(word["labels"][0]))
    #         word["labels"] = [Label("CONJ")]
    # print(sentence_dict)

    predicates = [
        "I am red",
        "It is red",
        "The index is less than `self.len()`",
        "The index is small",
        "Returns `true` if and only if `self == 2^k` for some `k`"
    ]

    prod_rules = {
        "OBJ": [
            # "I", "the index"
            ["PRP"],
            ["DT", "NN"],
            ["CODE"]
        ],
        "REL": [
            # less than `self.len()`.
            ["JJR", "IN", "OBJ"]
        ],
        # Property of an object
        "PROP": [
            # "am red", "is red"
            ["VBP", "JJ"],
            ["VBZ", "JJ"],
            # is (less than `self.len()`)
            ["VBZ", "REL"]
        ],
        # Object has property
        "ASSERT": [
            # "I am red", "It is red", "The index is small"
            ["OBJ", "PROP"],
        ],
        # If object has property
        "PRED": [
            ["IN", "ASSERT"],
            ["IFF", "ASSERT"]
        ],
        "IFF": [
            # if and only if
            ["IN", "CC", "RB", "IN"]
        ],
        "EXIST": [
            # "for some value"
            ["IN", "DT", "OBJ"]
        ],
        "RETIF": [
            # return obj if predicate
            ["RET", "OBJ", "PRED"]
        ],
        "IFRET": [
            ["PRED", "RET", "OBJ"]
        ]
    }

    productions = []
    for name, prods in prod_rules.items():
        productions.append(ProductionRule(name, prods))

    code_grammar = nltk.data.load("file:codegrammar.cfg")
    rd_parser = nltk.RecursiveDescentParser(code_grammar)
    for sent in predicates:
        sentence = Sentence(sent, use_tokenizer=CodeTokenizer())
        res = tagger.predict(sentence)
        token_dict = sentence.to_dict(tag_type="pos")
        correct_tokens(token_dict)

        labels = [x["labels"][0] for x in token_dict["entities"]]
        # print(sent)
        # print(labels)
        # print(extract_all_trees(productions, labels))

        nltk_sent = [l.value for l in labels]
        print(nltk_sent)
        for tree in rd_parser.parse(nltk_sent):
            print(tree)

# if __name__ == '__main__':
#     print(CodeTokenzer().tokenize("this is a code segment: `hello world`."))

    # for some k
