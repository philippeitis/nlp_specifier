from typing import Optional

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

nlp = spacy.blank('en')


def matcher_with_rule(name, rule):
    matcher = Matcher(nlp.vocab)
    if isinstance(rule[0], list):
        matcher.add(name, rule)
    else:
        matcher.add(name, [rule])
    return matcher


def lemma(word: str):
    return {"LEMMA": word.lower()}


def tag(tag: str):
    return {"TAG": tag.upper()}


def lower(text: str):
    return {"LOWER": text}


RET_LEMMA = {'LEMMA': "return"}
IS_OBJ = {
    "TAG": {"REGEX": "^(CODE)|(LIT)|(NN.*)$"}
}
ANY_TEXT = {"TEXT": {"REGEX": ".*"}}

RET_RULES = [
    [{'LEMMA': "return", "POS": "VERB"}],
    [RET_LEMMA, tag("DT"), tag("JJ"), tag("IN"), IS_OBJ],
    [RET_LEMMA, IS_OBJ],
    [RET_LEMMA, ANY_TEXT, IS_OBJ],
    [IS_OBJ, tag("VBZ"), RET_LEMMA],
    [IS_OBJ, RET_LEMMA],
]

ARITH = {
    "LOWER": {"IN": [
        "add", "plus", "subtract", "sub", "divide", "div", "multiply", "mul", "remainder",
        "rem", "added", "subtracted", "divided", "multiplied", "xor", "modulo",
    ]
    }
}

ARITH_SIGN = {
    "ORTH": {"IN": ["%", "+", "-", "/", "*"]}
}


def ret_rule_to_matcher(rule):
    for i, sub_rule in enumerate(rule):
        if "LEMMA" in sub_rule and sub_rule["LEMMA"] == "return":
            matcher = Matcher(nlp.vocab)
            matcher.add("RETURN", [rule])
            return i, {"tag_": "RET", "pos_": "VERB"}, matcher
    raise ValueError("NOT A RET RULE")


CODE_MATCHER = matcher_with_rule("CODE", [{'ORTH': "`"}, {'OP': '+'}, {'ORTH': "`"}])
STR_MATCHER = matcher_with_rule("STR", [{'ORTH': "\""}, {'OP': '*'}, {'ORTH': "\""}])
CHAR_MATCHER = matcher_with_rule("CHAR", [{'ORTH': "'"}, {"IS_ASCII": True, 'LENGTH': 1}, {'ORTH': "'"}])
SOME_MATCHER = matcher_with_rule("SOME", [{"TEXT": "Some"}, {'ORTH': "("}, {"OP": "+"}, {'ORTH': ")"}])
LIFETIME_MATCHER = matcher_with_rule("LIFETIME", [{"ORTH": "'"}, {"IS_ASCII": True}])
REF_MATCHER = matcher_with_rule("REF", [{"ORTH": "&"}, {"TAG": "LIFETIME", "OP": "?"}, {"ORTH": "mut", "OP": "?"}, IS_OBJ])
GENERIC_PATH_MATCHER = matcher_with_rule("GPATH", [
    [
        {"TEXT": {"REGEX": "^(::)?[a-zA-Z_][a-zA-Z0-9_]*(::[a-zA-Z_][a-zA-Z0-9_]*)*::<[a-zA-Z]+$"}},
        {"ORTH": ">"}
    ],
    [
        {"TEXT": {"REGEX": "^(::)?[a-zA-Z_][a-zA-Z0-9_]*(::[a-zA-Z_][a-zA-Z0-9_]*)*::<$"}},
        {"TAG": "REF", "OP": "?"},

BOOL_OPS = [
    ({"POS": "NOUN", "TAG": "BOOL_OP"}, matcher_with_rule(op, merge_bool_op(op)))
    for op in ["!=", "==", "&&", "||"]
]

FN_CALL = matcher_with_rule("CALL", [
    [{"TAG": "PATH"}, {"ORTH": "("}, IS_OBJ | {"OP": "?"}, {"ORTH": ")"}],
    [{"TAG": "GPATH"}, {"ORTH": "("}, IS_OBJ | {"OP": "?"}, {"ORTH": ")"}],
    [{"IS_ASCII": True}, {"ORTH": "("}, IS_OBJ | {"OP": "?"}, {"ORTH": ")"}]

])

WORD_MATCHERS_0 = [(idx, tag, matcher_with_rule(tag["tag_"], rule)) for idx, tag, rule in [
    (0, {"tag_": "PATH"}, [{"TEXT": {"REGEX": "^(::)?[a-zA-Z_][a-zA-Z0-9_]*(::[a-zA-Z_][a-zA-Z0-9_]*)+$"}}]),
]]

MERGE_MATCHERS = [
    ({"POS": "NOUN", "TAG": "CODE"}, CODE_MATCHER),
    ({"POS": "NOUN", "TAG": "STR"}, STR_MATCHER),
    ({"POS": "NOUN", "TAG": "CHAR"}, CHAR_MATCHER),
    ({"POS": "NOUN", "TAG": "OPTION"}, SOME_MATCHER),
    ({"POS": "NOUN", "TAG": "LIFETIME"}, LIFETIME_MATCHER),
    ({"POS": "NOUN", "TAG": "REF"}, REF_MATCHER),
    ({"POS": "NOUN", "TAG": "GPATH"}, GENERIC_PATH_MATCHER),
    ({"POS": "NOUN", "TAG": "CALL"}, FN_CALL)
]


WORD_MATCHERS = [(idx, tag, matcher_with_rule(tag["tag_"], rule)) for idx, tag, rule in [
    (0, {"tag_": "IF"}, [lemma("if")]),
    (0, {"tag_": "FOR"}, [lemma("for"), tag("DT")]),
    (1, {"tag_": "SHIFT"}, [IS_OBJ, lemma("shift"), tag("IN"), tag("DT"), tag("NN"), tag("IN"), IS_OBJ]),
    (1, {"tag_": "SHIFT"}, [IS_OBJ, tag("VBZ"), lemma("shift"), tag("IN"), tag("DT"), tag("NN"), tag("IN"), IS_OBJ]),
    (2, {"tag_": "SHIFT"}, [IS_OBJ, {"TAG": {"REGEX": "^JJ.*$"}}, lemma("shift"), IS_OBJ]),
    (1, {"tag_": "ARITH"}, [IS_OBJ, ARITH, IS_OBJ]),
    (1, {"tag_": "ARITH"}, [IS_OBJ, ARITH, tag("IN"), IS_OBJ]),
    (1, {"tag_": "ARITH"}, [IS_OBJ, ARITH_SIGN, IS_OBJ]),
    (1, {"tag_": "ARITH"}, [IS_OBJ, ARITH_SIGN, tag("IN"), IS_OBJ]),

    (0, {"tag_": "DOT"}, [lower(".")]),
    (0, {"tag_": "COMMA"}, [lower(",")]),
    (0, {"tag_": "EXCL"}, [lower("!")]),
    (0, {"tag_": "LIT"}, [{"LOWER": {"IN": ["true", "false"]}}]),
    (0, {"tag_": "ENCODING"}, [{"TEXT": {"REGEX": "^(?i)UTF(_|-)?(8|16)$"}}]),

]] + [ret_rule_to_matcher(rule) for rule in RET_RULES]


def get_literal_tag(word: str, idents=None) -> Optional[str]:
    """Determines whether `word` matches the pattern for a code literal - specifically,
     if the word is a Rust literal or a function argument.

    :param word:
    :param idents:
    :return:
    """
    # detect bool literals
    if idents and word in idents:
        return "IDENT"

    # detect numbers
    if word.endswith(("u8", "u16", "u32", "u64", "usize")):
        return "LIT_UNUM"

    if word.endswith(("i8", "i16", "i32", "i64", "isize")):
        return "LIT_INUM"

    if word.endswith(("f32", "f64")):
        return "LIT_FNUM"

    if word.isnumeric():
        return "LIT_FNUM" if "." in word else "LIT_INT"

    return None


def fix_tokens(doc: Doc, idents=None):
    for i, token in enumerate(doc):
        if get_literal_tag(token.text, idents):
            token.tag_ = "LIT"

    for idx, substitute, matcher in WORD_MATCHERS_0:
        for _, start, end in matcher(doc):
            for attr, val in substitute.items():
                setattr(doc[start + idx], attr, val)

    for attrs, matcher in MERGE_MATCHERS:
        while True:
            try:
                with doc.retokenize() as retokenizer:
                    _, start, end = next(iter(matcher(doc)))
                    retokenizer.merge(doc[start:end], attrs=attrs)
            except StopIteration:
                break

    for idx, substitute, matcher in WORD_MATCHERS:
        for _, start, end in matcher(doc):
            for attr, val in substitute.items():
                setattr(doc[start + idx], attr, val)
