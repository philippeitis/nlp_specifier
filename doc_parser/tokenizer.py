from typing import List

from flair.data import Token, Tokenizer
from doc_parser_utils import tokens_from_str


class CodeTokenizer(Tokenizer):
    def tokenize(self, sentence: str) -> List[Token]:
        return tokens_from_rust(sentence)


def tokens_from_rust(s: str) -> List[Token]:
    def rs_token_converter(token):
        return Token(token.text, token.index, whitespace_after=token.whitespace_after, start_position=token.start)
    return [rs_token_converter(token) for token in tokens_from_str(s)]


def repr_tokens(tokens):
    return " ".join(f"Token({t.text}, {t.idx}, {t.whitespace_after}, {t.start_position})" for t in tokens)


if __name__ == '__main__':
    test_cases = [
        "hello world!",
        "hello `self`!",
        "hello \"string literal\"!",
        "hello \"string literal\" aaaa !",
        "hello \"string literal\" aaaa \"!",
    ]

    for case in test_cases:
        print(case.find("!"))
        print(repr_tokens(tokens_from_rust(case)))
