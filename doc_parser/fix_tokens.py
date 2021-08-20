from typing import Optional, Tuple

from flair.data import Label

from lemmatizer import lemmatize


def get_label(t) -> str:
    return t["labels"][0].value


def set_label(t, label: str):
    t["labels"][0] = Label(label)


def text(t) -> str:
    return t["text"]


def set_text(t, word: str):
    t["text"] = word


def is_obj(t: str) -> bool:
    return t in {
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

    for i, entity in enumerate(entities):
        if i == 0:
            continue

        if is_obj(get_label(entity)):
            delta = len(entities) - i - 1
            if delta >= 6 and is_obj(get_label(entities[i + 6])):
                # SHIFT IN DT NN IN
                if lemmatize(text(entities[i + 1]).lower(), "v") == "shift" \
                        and get_label(entities[i + 2]) == "IN" \
                        and get_label(entities[i + 3]) == "DT" \
                        and get_label(entities[i + 4]).startswith("NN") \
                        and get_label(entities[i + 5]) == "IN":
                    set_label(entities[i + 1], "SHIFT")
            if delta >= 3 and is_obj(get_label(entities[i + 3])):
                # JJ SHIFT
                if get_label(entities[i + 1]).startswith("JJ") \
                        and lemmatize(text(entities[i + 2]).lower(), "v") == "shift":
                    set_label(entities[i + 2], "SHIFT")
                # ARITH IN
                elif is_arith(entities[i + 1]) \
                        and get_label(entities[i + 2]) == "IN":
                    set_label(entities[i + 1], "ARITH")
            # ARITH
            if delta >= 2 and is_obj(get_label(entities[i + 2])):
                if is_arith(entities[i + 1]) or text(entities[i + 1]).lower() == "xor":
                    set_label(entities[i + 1], "ARITH")


def fix_tokens(token_dict, idents=None):
    entities = token_dict["entities"]

    for i, entity in enumerate(entities):
        lit_tag, word = get_literal_tag(text(entity), idents)
        if lit_tag:
            set_label(entity, "LIT")
            set_text(entity, word)

    for i, entity in enumerate(entities):
        if text(entity).startswith("`"):
            entity["labels"][0] = Label("CODE")
            continue

        try:
            word = lemmatize(text(entity).lower(), get_label(entity))
        except ValueError:
            continue

        if word == "return":
            if get_label(entity).startswith("VB"):
                set_label(entity, "RET")
            elif i + 1 < len(entities) and is_obj(get_label(entities[i + 1])):
                set_label(entity, "RET")
            elif i + 2 < len(entities) and is_obj(get_label(entities[i + 2])):
                set_label(entity, "RET")
            elif i >= 2 and is_obj(get_label(entities[i - 2])) and get_label(entities[i - 1]) in {"VBZ"}:
                set_label(entity, "RET")
            elif i >= 1 and is_obj(get_label(entities[i - 1])):
                set_label(entity, "RET")
        elif word == "if":
            set_label(entity, "IF")
        elif word == "for":
            if i + 1 < len(entities) and get_label(entities[i + 1]) in {"DT"}:
                set_label(entity, "FOR")

    apply_operation_tokens(token_dict)
