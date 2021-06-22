from typing import Union, Optional
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree import Tree

from code_split import replace_code


def is_code_punc(child: Union[str, Tree]) -> bool:
    if isinstance(child, str):
        return False
    return child[0] == "`"


def is_lbracket(child: Union[str, Tree]) -> bool:
    if isinstance(child, str) or len(child) != 1:
        return False
    return child[0] == "-LRB-" or is_lbracket(child[0])


def is_rbracket(child: Union[str, Tree]) -> bool:
    if isinstance(child, str) or len(child) != 1:
        return False
    return child[0] == "-RRB-" or is_rbracket(child[0])


def convert_brackets(tree: Union[str, Tree]):
    """Converts curly bracket `-LRB-` and `-RRB-` into `(` and `)`, respectively.

    :param tree: An instance of nltk.tree.Tree
    :return:
    """
    if isinstance(tree, str):
        return

    for i, child in enumerate(tree):
        if is_lbracket(child):
            tree[i] = Tree("LBRACKET", ["("])
        elif is_rbracket(child):
            tree[i] = Tree("RBRACKET", [")"])
        convert_brackets(child)


def convert_code_segments_helper(tree: Union[str, Tree], start: int = 0) -> Optional[int]:
    if isinstance(tree, str):
        return None

    for i in range(start, len(tree)):
        for j in range(i + 1, len(tree)):
            if is_code_punc(tree[i]) and is_code_punc(tree[j]):
                for x, item in enumerate(tree[i: j + 1]):
                    if is_lbracket(item):
                        tree[i + x] = '('
                    elif is_rbracket(item):
                        tree[i + x] = ')'
                    else:
                        convert_brackets(item)

                sub_tree = Tree("CODE", tree[i:j + 1])
                for _ in range(0, j - i + 1):
                    del tree[i]
                tree.insert(i, sub_tree.flatten())
                return i + 1
    return None


def convert_code_segments(parse_tree: Union[str, Tree]) -> Optional[Tree]:
    if isinstance(parse_tree, str):
        return

    for i, tree in enumerate(parse_tree):
        index = 0
        while index is not None:
            index = convert_code_segments_helper(tree, index)

        # if is_code_segment(child):
        #     child.set_label("CODE")
        #     parse_tree[i] = child.flatten()
        convert_code_segments(tree)
    return parse_tree


class Parser:
    def __init__(self):
        self.nlp = CoreNLPParser(url='http://localhost:9000')
        self.properties = {
            'annotators': 'parse',
            'outputFormat': 'json',
            'timeout': 1000,
        }

    def parse_sentence(self, text: Union[str, Tree]) -> Tree:
        return convert_code_segments(next(self.nlp.raw_parse(text, properties=self.properties)))


class Specification:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.scan_conditionals(self.tree)

    @classmethod
    def conditional_index(cls, tree: Union[str, Tree]):
        if isinstance(tree, str):
            return
        for i, child in enumerate(tree):
            if isinstance(child, str):
                continue
            if child.label() in {"SBAR", "SINV", "SQ"}:
                return i
        return

    @classmethod
    def scan_conditionals(cls, tree: Union[str, Tree]):
        if isinstance(tree, str):
            return
        for child in tree:
            index = cls.conditional_index(child)
            if index is None:
                pass
            cls.scan_conditionals(child)
# Terms: http://erwinkomen.ruhosting.nl/eng/2014_Longdale-Labels.htm
# Note: First word seems to be treated as noun for some reason (eg. Returns, Shifts)

# Fragment containing SBAR at end, or S fragment


def split_out_code(sentence: str) -> [str]:
    splits = sentence.split("`")
    # find `, scan forwards for next `, and split
    parts = []
    i = 0
    j = 1
    while i < len(sentence):
        if sentence[i] == "`":
            while j < len(sentence):
                if sentence[j] != "`":
                    parts.append(sentence)
            for j in range(i + 1, len(sentence)):
                if sentence[j] == "`":
                    parts.append(sentence[:i])
                    parts.append(sentence[i+1:j])
                    i = j
                    break
        i += 1
    parts.append(sentence[i:j])


if __name__ == '__main__':
    parser = Parser()

    # Eg. Identify what this does as a simple description.
    return_style_sentences = []
    if_style_sentences = []

    # returns x if this otherwise y
    # if this, returns x, otherwise y
    # returns y if this. if that, returns x.
    # returns x when this. when that, returns y.
    # if this and that, returns x, otherwise y.

    # Notes: Need to identify when functions should be trusted, pure, or smth else.
    # (Should fn's be pure if no side-effectful fns? (eg. no networking, no syscalls, no asm, no unsafe, and no mutability.)
    sentences = [
        "The result is not equal to one."
        # "Computes `self - rhs`, returning `None` if overflow occurred.",
        # "Computes `self - rhs`.",
        # "If overflow occurs, returns `None`.",
        # "For each item in the list, if it is False, increment the count by one.",
        # "For each `item` in the list, if `item.empty()` is `True`, increment the count by one.",
        # "If `abcdef[0]`, increment the count by one.",
        # "If `abcdef[0] == True`, increment the count by one.",
        # "If `abcdef[0]` is `True`, increment the count by one.",  # <- quotes at different levels
        # """Please note this isn’t the same operation as the `<<` shifting operator!""",
        # "Returns_VB a BitVec32.",
        # "For each index `i` from 0, up to and not including 32, the value at index `i` in the result is false.",
        # """Returns true, if the object is brown, otherwise false."""
    ]

    for sentence in sentences:
        s, subs = replace_code(sentence)
        sentence = parser.parse_sentence(s)
        print(sentence)
        Specification(sentence)

"""
Returns_VB the number of leading zeros in the binary representation of self.
Returns the number of leading ones in the binary representation of self.
Returns the number of trailing ones in the binary representation of self.
Shifts the bits to the left by a specified amount, n, wrapping the truncated bits to the end of the resulting integer.
Please note this isn’t the same operation as the `<<` shifting operator!

Converts `self` to big endian from the target’s endianness.
On big endian this is a no-op. On little endian the bytes are swapped.
Checked integer subtraction. Computes `self - rhs`, returning `None` if overflow occurred ().
"""


"""
NP VP
(obj) (action)
"""
"""
is equal to
is not equal to
does not equal
(cmp)

"""

" if the index is equal to `self.len()`"

