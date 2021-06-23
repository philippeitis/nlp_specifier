from enum import Enum, auto
from typing import List, Union

import flair.data
from flair.data import Sentence, Label, Token
from flair.models import MultiTagger

from nltk.tree import Tree
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
                sind = sentence[a:].find(" ")
                cind = sentence[a:].find(",")
                if cind == -1:
                    ind = sind
                elif sind == -1:
                    ind = cind
                else:
                    ind = min(sind, cind)
            if ind == -1:
                t = Token(sentence[a:], num, whitespace_after=False, start_position=a)
                parts.append(t)
                break
            else:
                t = Token(sentence[a: a + ind], num, whitespace_after=True, start_position=a)
                parts.append(t)

            num += 1
            a = a + ind
            while a < len(sentence) and sentence[a] in {",", " "}:
                # if sentence[a] == ",":
                #     t = Token(sentence[a], num, whitespace_after=False, start_position=a)
                #     parts.append(t)
                #     num += 1

                a += 1

        return parts


def print_tokens(token_dict):
    for entity in token_dict["entities"]:
        print(entity["text"], entity["labels"][0])
    print()


def might_be_code(s: str):
    # detect bool literals
    if s.lower() in {"true", "false", "self"}:
        return True

    # detect numbers
    if s.endswith(("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64")):
        return True

    if s.isnumeric():
        return True

    return False


def correct_tokens(token_dict):
    entities = token_dict["entities"]

    for entity in entities:
        if entity["text"].startswith("`"):
            entity["labels"][0] = Label("CODE")

    for i, entity in enumerate(entities):
        if might_be_code(entity["text"]):
            entity["labels"][0] = Label("LIT")

    if len(entities) <= 2:
        return

    for i, entity in enumerate(entities):
        if entity["text"].lower() == "returns":
            if i + 1 < len(entities) and entities[i + 1]["labels"][0].value in {
                "NN", "NNP", "NNPS", "NNS", "CODE", "LIT"
            }:
                entity["labels"][0] = Label("RET")

    for i, entity in enumerate(entities):
        if entity["text"].lower() == "for":
            if i + 1 < len(entities) and entities[i + 1]["text"].lower() in {
                "each"
            }:
                entity["labels"][0] = Label("FOR")
                entities[i + 1]["labels"][0] = Label("EACH")


class Code:
    def __init__(self, tree: Tree):
        self.code: str = tree[0]

    def __str__(self):
        return self.code.strip("`")


class Literal:
    def __init__(self, tree: Tree):
        self.str: str = tree[0]

    def __str__(self):
        return self.str


class Object:
    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in {("PRP",), ("DT", "NN"), ("CODE",), ("LIT",)}:
            raise ValueError(f"Bad tree - expected OBJ productions, got {labels}")
        self.labels = labels
        self.tree = tree

    def as_code(self):
        if self.labels == ("CODE",):
            return Code(self.tree[0])
        if self.labels == ("LIT",):
            return Literal(self.tree[0])
        if self.labels == ("DT", "NN"):
            # TODO: This assumes that val in "The val" is a variable.
            return self.tree[1][0]
        raise ValueError(f"{self.labels} not handled.")

    def is_output(self):
        # TODO: This should return true if self is "result" or "output".
        pass
        # TODO: implement method to cross-ref function inputs


class Comparator(Enum):
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    EQ = auto()
    NEQ = auto()

    @classmethod
    def from_str(cls, s: str):
        return {
            "less": cls.LT,
            "smaller": cls.LT,
            "greater": cls.GT,
            "larger": cls.GT,
            "equal": cls.EQ,
            "unequal": cls.NEQ,
            "same": cls.EQ,
        }[s.lower()]

    def __str__(self):
        return {
            self.LT: "<",
            self.GT: ">",
            self.LTE: "<=",
            self.GTE: ">=",
            self.EQ: "==",
            self.NEQ: "!=",
        }[self]

    def apply_eq(self):
        return {
            self.LT: self.LTE,
            self.GT: self.GTE,
        }.get(self, self)

    def negate(self, negate):
        if not negate:
            return self

        return {
            self.LT: self.GTE,
            self.GT: self.LTE,
            self.LTE: self.GT,
            self.GTE: self.LT,
            self.EQ: self.NEQ,
            self.NEQ: self.EQ,
        }[self]


class TJJ:
    def __init__(self, tree: Tree):
        self.tree = tree

    def get_word(self):
        return self.tree[-1][0]


class Relation:
    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in {("TJJ", "IN", "OBJ"), ("TJJ", "EQTO", "OBJ")}:
            raise ValueError(f"Bad tree - expected REL productions, got {labels}")
        self.labels = labels
        self.tree = tree
        self.negate = False

    def as_code(self):
        relation = Comparator.from_str(TJJ(self.tree[0]).get_word()).negate(self.negate)
        if self.labels == ("TJJ", "EQTO", "OBJ"):
            relation = relation.apply_eq()
        return f" {relation} {Object(self.tree[2]).as_code()}"

    def apply_adverb(self, s: str):
        if s.lower() != "not":
            raise ValueError(f"Only not is supported as an adverb for Relation: got {s}")
        self.negate = True
        return self

    def set_negate(self, negate: bool):
        self.negate = negate
        return self


class ModRelation(Relation):
    def __init__(self, tree: Tree):
        if tree[0].label() == "RB":
            super().__init__(tree[1])
            self.apply_adverb(tree[0][0])
        else:
            super().__init__(tree[0])


class MVB:
    def __init__(self, tree: Tree):
        self.tree = tree

    def is_negation(self):
        return self.tree[0].label() == "RB" and self.tree[0][0].lower() == "not"


class Property:
    PRODUCTIONS = {
        ("MVB", "JJ"),
        ("MVB", "MREL"),
        ("MVB", "LIT"),
    }

    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in self.PRODUCTIONS:
            raise ValueError(f"Bad tree - expected PROP productions, got {labels}")
        self.labels = labels
        self.tree = tree
        self.negate = MVB(self.tree[0]).is_negation()

    def as_code(self, lhs: Object):
        if self.labels[-1] == "JJ":
            sym = "!" if self.negate else ""
            return f"{sym}{lhs.as_code()}.{self.tree[1][0]}()"
        if self.labels[-1] == "LIT":
            cmp = Comparator.EQ.negate(self.negate)
            return f"{lhs.as_code()} {cmp} {self.tree[1][0]}"
        if self.labels[-1] == "MREL":
            return f"{lhs.as_code()}{ModRelation(self.tree[-1]).set_negate(MVB(self.tree[0]).is_negation()).as_code()}"

        raise ValueError(f"Case {self.labels} not handled.")


class Assert:
    def __init__(self, tree: Tree):
        if tree[0].label() != "OBJ":
            raise ValueError(f"Bad tree - expected OBJ, got {tree[0].label()}")
        if tree[1].label() != "PROP":
            raise ValueError(f"Bad tree - expected PROP, got {tree[1].label()}")

        self.obj = Object(tree[0])
        self.prop = Property(tree[1])

    def as_code(self):
        return f"{self.prop.as_code(self.obj)}"


class HardAssert(Assert):
    def __init__(self, tree: Tree):
        if tree[1].label() != "MD":
            raise ValueError(f"Bad tree - expected MD as first HASSERT production, got {tree[0].label()}")
        super().__init__([tree[0], tree[2]])
        self.md = tree[1]

    def is_precondition(self):
        # TODO: This should be more rigourous:
        #  eg. The [result] must be some value
        #  vs. The input must be some value.
        #  Check what MD is in relation to.
        #  if self.obj.is_input() // if self.obj.is_output()
        # will indicates future
        return self.md[0] not in {"will"}

    def as_spec(self):
        if self.is_precondition():
            cond = "requires"
        else:
            cond = "ensures"
        return f"#[{cond}({self.as_code()})]"


class Range:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.labels = tuple(x.label() for x in tree)

    def as_foreach_pred(self):
        if self.labels == ("NN", "OBJ", "IN", "OBJ", "IN", "OBJ"):
            ident = Object(self.tree[1])
            start = Object(self.tree[3])
            end = Object(self.tree[5])

            return ident, start, end
        raise ValueError(f"Range case not handled: {self.labels}")


class RangeMod(Enum):
    INCLUSIVE = auto()
    EXCLUSIVE = auto()

    @classmethod
    def from_str(cls, s: str):
        return {
            "inclusive": cls.INCLUSIVE,
            "exclusive": cls.EXCLUSIVE
        }[s.lower()]

    def as_code_lt(self):
        return {
            self.INCLUSIVE: "<=",
            self.EXCLUSIVE: "<"
        }[self]


class ForEach:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.range = Range(tree[2])
        self.labels = tuple(x.label() for x in tree)

    def as_code(self, cond: str):
        # need type hint about second
        ident, start, end = self.range.as_foreach_pred()

        if self.labels[-1] == "JJ":
            upper_range = RangeMod.from_str(self.tree[-1][0]).as_code_lt()
        else:
            upper_range = RangeMod.EXCLUSIVE.as_code_lt()
        # type hints: start.as_code(), end.as_code()
        #                                    v do type introspection here
        xtype = "int"
        # detect multiple identifiers.
        return f"forall(|{ident.as_code()}: {xtype}| ({start.as_code()} <= {ident.as_code()} && {ident.as_code()} {upper_range} {end.as_code()}) ==> ({cond}))"
        # raise ValueError(f"ForEach case not handled: {self.labels}")


class ForEachHardAssert:
    def __init__(self, tree: Tree):
        self.foreach = ForEach(tree[0])
        self.hassert = HardAssert(tree[-1])

    def as_spec(self):
        if self.hassert.is_precondition():
            cond = "requires"
        else:
            cond = "ensures"
        return f"#[{cond}({self.foreach.as_code(self.hassert.as_code())})]"


class Predicate:
    def __init__(self, tree: Tree):
        if tree[0].label() not in {"IFF", "IN"}:
            raise ValueError(f"Bad tree - expected IFF or IN, got {tree[0].label()}")
        if tree[1].label() != "ASSERT":
            raise ValueError(f"Bad tree - expected ASSERT, got {tree[1].label()}")

        self.iff = tree[0].label() == "IFF"
        self.assertion = Assert(tree[1])

    def as_code(self):
        return self.assertion.as_code()


class ReturnIf:
    def __init__(self, tree: Union[Tree, List[Tree]]):
        if tree[0].label() != "RET":
            raise ValueError(f"Bad tree - expected RET, got {tree[0].label()}")

        if tree[1].label() != "OBJ":
            raise ValueError(f"Bad tree - expected OBJ, got {tree[1].label()}")

        if tree[2].label() not in {"EPRED", "PRED"}:
            raise ValueError(f"Bad tree - expected EPRED or PRED, got {tree[2].label()}")

        # Need to find out what the ret value is.

        self.ret_val = Object(tree[1])
        self.pred = Predicate(tree[2])

    def __str__(self):
        return f"return {self.ret_val.as_code()};"

    def as_spec(self):
        if self.pred.iff:
            return f"""#[ensures({self.pred.as_code()} ==> (result == {self.ret_val.as_code()}))]
#[ensures((result == {self.ret_val.as_code()}) ==> {self.pred.as_code()})]"""
        return f"""#[ensures({self.pred.as_code()} ==> (result == {self.ret_val.as_code()}))]"""


class IfReturn(ReturnIf):
    def __init__(self, tree: Tree):
        # Swizzle the values as needed.
        if tree[1].label() == "COMMA":
            super().__init__([tree[2], tree[3], tree[0]])
        else:
            super().__init__([tree[1], tree[2], tree[0]])


class Existential:
    def __init__(self, tree: Tree):
        pass


class Specification:
    def __init__(self, tree: Tree):
        self.spec = {
            "RETIF": ReturnIf,
            "IFRET": IfReturn,
            "HASSERT": HardAssert,
            "UHASSERT": ForEachHardAssert,
        }[tree[0].label()](tree[0])

    def as_spec(self):
        return self.spec.as_spec()


def get_bottom_nodes(tree: Tree, nodes: List[Tree]):
    if isinstance(tree, str):
        return nodes

    if len(tree) == 1 and isinstance(tree[0], str):
        nodes.append(tree)
    else:
        for sub_tree in tree:
            get_bottom_nodes(sub_tree, nodes)

    return nodes


def attach_words_to_nodes(tree: Tree, words: List[str]):
    nodes = get_bottom_nodes(tree, [])
    for node, word in zip(nodes, words):
        node[0] = word

    return tree


class POSModel(Enum):
    POS = auto()
    POS_FAST = auto()
    UPOS = auto()
    UPOS_FAST = auto()

    def __str__(self):
        return {
            POSModel.POS: "pos",
            POSModel.POS_FAST: "pos-fast",
            POSModel.UPOS: "upos",
            POSModel.UPOS_FAST: "upos-fast"
        }[self]


class Parser:
    def __init__(self, pos_model: POSModel = POSModel.POS, grammar_path: str = "doc_parser/codegrammar.cfg"):
        self.root_tagger = MultiTagger.load([str(pos_model)])
        self.tokenizer = CodeTokenizer()

        self.grammar = nltk.data.load(f"file:{grammar_path}")
        self.rd_parser = nltk.RecursiveDescentParser(self.grammar)

    def parse_sentence(self, sentence: str):
        sentence = sentence \
            .replace("isn't", "is not") \
            .rstrip(".")

        sentence = Sentence(sentence, use_tokenizer=self.tokenizer)
        self.root_tagger.predict(sentence)
        token_dict = sentence.to_dict(tag_type="pos")
        correct_tokens(token_dict)
        labels = [entity["labels"][0] for entity in token_dict["entities"]]
        words = [entity["text"] for entity in token_dict["entities"]]
        print(labels)
        print(words)
        nltk_sent = [label.value for label in labels]
        return [
            attach_words_to_nodes(tree, words)
            for tree in self.rd_parser.parse(nltk_sent)
        ]


def main():
    predicates = [
        "I am red",
        "It is red",
        "The index is less than `self.len()`",
        "The index is 0u32",
        # Well handled.
        "Returns `true` if and only if `self` is 0u32",
        "Returns `true` if `self` is 0u32",
        "Returns true if self is 0u32",
        # Need to find a corresponding function or method for this case.
        "Returns `true` if and only if `self` is blue",
        # Well handled.
        "Returns `true` if the index is greater than or equal to `self.len()`",
        "Returns `true` if the index is equal to `self.len()`",
        "Returns `true` if the index is not equal to `self.len()`",
        "Returns `true` if the index isn't equal to `self.len()`.",
        "Returns `true` if the index is not less than `self.len()`",
        "Returns `true` if the index is smaller than or equal to `self.len()`",
        "If the index is smaller than or equal to `self.len()`, returns true.",
        "`index` must be less than `self.len()`",
        "for each index `i` from `0` to `self.len()`, `i` must be less than `self.len()`",
        "for each index `i` from `0` to `self.len()` inclusive, `i` must be less than `self.len()`",
        "`i` must be less than `self.len()`",
        "for each index `i` from `0` to `self.len()` inclusive, `i` must be less than or equal to `self.len()`",
        "for each index `i` from `0` to `self.len()`, `self.lookup(i)` must not be equal to 0",
        # differentiate btw will and must wrt result
        "for each index `i` from `0` to `self.len()`, `self.lookup(i)` will not be the same as `result.lookup(i)`"

    ]

    parser = Parser()
    for sentence in predicates:
        print("=" * 80)
        print("Sentence:", sentence)
        print("=" * 80)
        for tree in parser.parse_sentence(sentence):
            tree: nltk.tree.Tree = tree
            try:
                spec = Specification(tree)
                print(spec.as_spec())
            except LookupError as e:
                print(f"No specification found. {e}")
            print(tree)
            print()


if __name__ == '__main__':
    main()
