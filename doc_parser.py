from typing import List

import flair.data
from flair.data import Sentence, Label, Token
from flair.models import MultiTagger

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


class Assert:
    pass


class Parser:
    def __init__(self, pos_tagger: str = "pos", grammar_path: str = "codegrammar.cfg"):
        self.root_tagger = MultiTagger.load([pos_tagger])
        self.grammar = nltk.data.load(f"file:{grammar_path}")
        self.rd_parser = nltk.RecursiveDescentParser(self.grammar)
        self.tokenizer = CodeTokenizer()

    def parse_sentence(self, sentence: str):
        sentence = Sentence(sentence, use_tokenizer=self.tokenizer)
        self.root_tagger.predict(sentence)
        token_dict = sentence.to_dict(tag_type="pos")
        correct_tokens(token_dict)
        labels = [entity["labels"][0] for entity in token_dict["entities"]]

        nltk_sent = [label.value for label in labels]

        return [tree for tree in self.rd_parser.parse(nltk_sent)]


def main():
    predicates = [
        "I am red",
        "It is red",
        "The index is less than `self.len()`",
        "The index is small",
        "Returns `true` if and only if `self == 2^k` for some `k`"
    ]

    parser = Parser()
    for sentence in predicates:
        print(sentence)
        for tree in parser.parse_sentence(sentence):
            print(tree)


if __name__ == '__main__':
    main()
