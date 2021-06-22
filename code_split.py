from typing import Dict


def split_code(sentence: str) -> [(str, bool)]:
    parts = []
    a = 0
    b = 0
    open_quote = False
    while b < len(sentence):
        if sentence[b] == "`":
            parts.append((sentence[a:b], open_quote))
            open_quote = not open_quote
            a = b + 1
        b += 1
    parts.append((sentence[a:], False))

    return parts


def replace_code(sentence: str) -> (str, Dict[str, str]):
    s = ""
    substitutions = {}
    i = 0
    for (part, is_code) in split_code(sentence):
        if is_code:
            s += f"CODE{i}"
            substitutions[f"CODE{i}"] = part
            i += 1
        else:
            s += part
    return s, substitutions


if __name__ == "__main__":
    print(split_code("this is a code segment: `hello world`."))
    print(replace_code("this is not a code segment: `hello world."))
    print(replace_code("this is two code segments: `hello world`, `yeehaw`."))
