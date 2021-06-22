from typing import List

import flair.data
from flair.data import Sentence, Label, Token
from flair.models import MultiTagger


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
            if i + 1 < len(entities) and entities[i+1]["labels"][0].value in {"NN", "NNP", "NNPS", "NNS", "CODE"}:
                entity["labels"][0] = Label("VBZ")


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
    for sent in predicates:
        sentence = Sentence(sent, use_tokenizer=CodeTokenizer())
        res = tagger.predict(sentence)
        token_dict = sentence.to_dict(tag_type="pos")
        correct_tokens(token_dict)
        print_tokens(token_dict)
    prod_rules = {
        "PRED": [
            # "I am red"
            ["PRP", "VBP", "JJ"],
            # "It is red"
            ["PRP", "VBZ", "JJ"],
            # "The index is less than `self.len()`"
            ["DT", "NN", "VBZ", "JJR", "IN", "``", "NN", "."],
            # "The index is small"
            ["DT", "NN", "VBZ", "JJ"],

        ]

    }


# if __name__ == '__main__':
#     print(CodeTokenzer().tokenize("this is a code segment: `hello world`."))
