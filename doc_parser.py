from copy import copy
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


if __name__ == '__main__':
    tagger = MultiTagger.load(['pos'])

    predicates = [
        "I am red",
        "It is red",
        "The index is less than `self.len()`",
        "The index is small",
        "Returns `true` if and only if `self == 2^k` for some `k`"
    ]

    code_grammar = nltk.data.load("file:codegrammar.cfg")
    rd_parser = nltk.RecursiveDescentParser(code_grammar)
    for sent in predicates:
        # Use FLAIR for more flexible root level POS tags
        sentence = Sentence(sent, use_tokenizer=CodeTokenizer())
        tagger.predict(sentence)
        # Correct tokens for words like "Return", which are not
        # handled correctly.
        token_dict = sentence.to_dict(tag_type="pos")
        correct_tokens(token_dict)

        # Extract labels for nltk CFG parser.
        labels = [x["labels"][0] for x in token_dict["entities"]]

        nltk_sent = [l.value for l in labels]

        # Parse trees here.
        for tree in rd_parser.parse(nltk_sent):
            tree: nltk.tree.Tree = tree
            print(tree)
            tree.label()
