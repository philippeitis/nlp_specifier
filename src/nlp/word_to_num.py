from typing import Tuple

from word2number import w2n

__all__ = ["EnglishNumber"]

BASE_WORDS = (
    ("first", 1),
    ("second", 2),
    ("third", 3),
    ("fifth", 5),
    ("eighth", 8),
    ("ninth", 9),
)

ENDINGS = ("st", "nd", "rd", "th")


def cardinal_word_to_num(word: str) -> Tuple[bool, int, bool]:
    word = word.lower()
    is_neg = word.startswith("minus") or word.startswith("negative")
    for target, bottom_num in BASE_WORDS:
        if word.endswith(target):
            stripped = word.removesuffix(target)
            if not stripped:
                return is_neg, bottom_num, True
            return is_neg, w2n.word_to_num(stripped) + bottom_num, True

    if word.endswith("ieth"):
        return is_neg, w2n.word_to_num(word.removesuffix("ieth") + "y"), True

    if word == "twelfth":
        return is_neg, 12, True

    for ending in ENDINGS:
        if word.endswith(ending):
            return is_neg, w2n.word_to_num(word.removesuffix(ending)), True

    return is_neg, w2n.word_to_num(word), False


class EnglishNumber:
    def __init__(self, word: str):
        is_neg, self.num, self.is_cardinal = cardinal_word_to_num(word)
        if is_neg:
            self.num = -self.num

    def __str__(self):
        return f"Number({self.is_cardinal}, {self.num})"


if __name__ == "__main__":
    cases = [
        ("first", 1, True),
        ("second", 2, True),
        ("third", 3, True),
        ("fourth", 4, True),
        ("fifth", 5, True),
        ("six", 6, False),
        ("sixth", 6, True),
        ("eighth", 8, True),
        ("ninth", 9, True),
        ("tenth", 10, True),
        ("eleventh", 11, True),
        ("twelfth", 12, True),
        ("thirteenth", 13, True),
        ("twentieth", 20, True),
        ("thirty-sixth", 36, True),
        ("twenty-fifth", 25, True),
        ("minus three", -3, False),
        ("three point five", 3.5, False),
    ]

    for word, expected_num, expected_cardinality in cases:
        en = EnglishNumber(word)
        assert en.num == expected_num, f"{en.num} != {expected_num} ({word})"
        assert (
            en.is_cardinal == expected_cardinality
        ), f"{en.is_cardinal} != {expected_cardinality} ({word})"
