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
        while a < len(sentence) and sentence[a].isspace():
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


def might_be_code(word: str, key_words=None) -> bool:
    # detect bool literals
    if word.lower() in {"true", "false", "self"}:
        return True

    if key_words and word in key_words:
        return True

    # detect numbers
    if word.endswith(("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64")):
        return True

    if word.isnumeric():
        return True

    return False


def correct_tokens(token_dict, key_words=None):
    entities = token_dict["entities"]

    for entity in entities:
        if entity["text"].startswith("`"):
            entity["labels"][0] = Label("CODE")

    for i, entity in enumerate(entities):
        if might_be_code(entity["text"], key_words):
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
            if i + 1 < len(entities) and entities[i + 1]["labels"][0].value in {"DT"}:
                entity["labels"][0] = Label("FOR")


class Code:
    def __init__(self, tree: Tree):
        self.code: str = tree[0]

    def as_code(self):
        return self.code.strip("`")


class Literal:
    def __init__(self, tree: Tree):
        self.str: str = tree[0]

    def as_code(self):
        return self.str


class Object:
    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in {("PRP",), ("DT", "MNN"), ("CODE",), ("LIT",), ("DT", "LIT"), ("MNN",)}:
            raise ValueError(f"Bad tree - expected OBJ productions, got {labels}")
        self.labels = labels
        self.tree = tree

    def as_code(self) -> str:
        if self.labels == ("CODE",):
            return Code(self.tree[0]).as_code()
        if self.labels[-1] == "LIT":
            return Literal(self.tree[-1]).as_code()
        if self.labels == ("DT", "MNN"):
            # TODO: This assumes that val in "The val" is a variable.
            return self.tree[1][0]
        if self.labels == ("MNN",):
            # TODO: This assumes that val in "The val" is a variable.
            return self.tree[0][0][0]

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
        if tree[0].label() == "REL":
            raise ValueError("Recursive MREL not supported")
        else:
            labels = tuple(x.label() for x in tree)
            if labels not in {("TJJ", "IN", "OBJ"), ("TJJ", "EQTO", "OBJ"), ("IN", "OBJ")}:
                raise ValueError(f"Bad tree - expected REL productions, got {labels}")
            self.labels = labels
            self.tree = tree
            self.negate = False

    def as_code(self, lhs: Object):
        relation = Comparator.from_str(TJJ(self.tree[0]).get_word()).negate(self.negate)
        if self.labels == ("TJJ", "EQTO", "OBJ"):
            relation = relation.apply_eq()
        return f"{lhs.as_code()} {relation} {Object(self.tree[2]).as_code()}"

    def apply_adverb(self, s: str):
        if s.lower() != "not":
            raise ValueError(f"Only not is supported as an adverb for Relation: got {s}")
        self.negate = True
        return self

    def apply_negate(self, negate: bool):
        if negate:
            self.negate = not self.negate
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

    def vb(self):
        return self.tree[-1][0]


class MJJ:
    def __init__(self, tree: Tree):
        self.mod = tree[0] if tree[0].label() == "RB" else None
        self.vb = tree[-1]

    # TODO: Expand tests here.
    def is_negative(self):
        return self.mod is not None and self.mod[0] == "not"

    # TODO: Expand tests for checking if a particular value is unchanged.
    def unchanged(self):
        if self.word() in {"unchanged"}:
            return True
        return False

    def word(self):
        return self.vb[0]


class Property:
    PRODUCTIONS = {
        ("MVB", "MJJ"),
        ("MVB", "MREL"),
        ("MVB", "OBJ"),
        ("MVB", "RANGEMOD"),
        ("MVB",)
    }

    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in self.PRODUCTIONS:
            raise ValueError(f"Bad tree - expected PROP productions, got {labels}")
        self.labels = labels
        self.tree = tree
        self.mvb = MVB(self.tree[0])
        self.negate = self.mvb.is_negation()

    # F F -> F no neg
    # F T -> T is not changed
    # T F -> T not is  changed -> is changed
    # T T -> F (if 1, neg is true, otherwise true)
    def as_code(self, lhs: Object):
        if self.labels[-1] == "MJJ":
            mjj = MJJ(self.tree[-1])
            neg = mjj.is_negative() != self.negate
            if mjj.unchanged():
                sym = "!" if neg else "="
                return f"{lhs.as_code()} {sym}= old({lhs.as_code()})"
            sym = "!" if neg else ""
            return f"{sym}{lhs.as_code()}.{mjj.word()}()"

        if self.labels[-1] == "MREL":
            return ModRelation(self.tree[-1]).apply_negate(self.negate).as_code(lhs)

        if self.labels[-1] == "OBJ":
            cmp = Comparator.EQ.negate(self.negate)
            return f"{lhs.as_code()} {cmp} {Object(self.tree[1]).as_code()}"

        if self.labels[-1] == "MVB":
            if self.mvb.vb() in {"changed", "modified", "altered", "change"}:
                if self.negate:
                    return f"{lhs.as_code()} == old({lhs.as_code()})"
                return f"{lhs.as_code()} != old({lhs.as_code()})"
            elif self.mvb.vb() in {"unchanged", "unmodified", "unaltered"}:
                if self.negate:
                    return f"{lhs.as_code()} != old({lhs.as_code()})"
                return f"{lhs.as_code()} == old({lhs.as_code()})"
            else:
                raise ValueError(f"PROP: unexpected verb in MVB case ({self.mvb.vb()})")

        if self.labels[-1] == "RANGEMOD":
            r = RangeMod(self.tree[-1])
            ident, start, stop = r.as_foreach_pred()
            if ident:
                raise ValueError(f"PROP: Unexpected ident in RANGE case: {ident}")
            return f"{start.as_code()} <= {lhs.as_code()} && {lhs.as_code()} {r.upper_bound} {stop.as_code()}"

        raise ValueError(f"Case {self.labels} not handled.")


class Assert:
    def __init__(self, tree: Tree):
        if tree[0].label() != "OBJ":
            raise ValueError(f"Bad tree - expected OBJ, got {tree[0].label()}")

        if tree[-1].label() == "ASSERT":
            if tree[1].label() != "CC":
                raise ValueError(f"Bad tree - expected PROP, got {tree[1].label()}")

            assertx = Assert(tree[-1])
            self.objs = assertx.objs + [(Object(tree[0]), CC(tree[1]))]
            self.prop = assertx.prop
        else:
            if tree[1].label() != "PROP":
                raise ValueError(f"Bad tree - expected PROP, got {tree[1].label()}")

            self.objs = [(Object(tree[0]), None)]
            self.prop = Property(tree[1])

    def as_code(self):
        if len(self.objs) == 1:
            return f"{self.prop.as_code(self.objs[0][0])}"

        conditions = [[f"{self.prop.as_code(self.objs[0][0])}"]]
        for obj, cc in self.objs[1:]:
            assert isinstance(cc, CC)
            if cc.bool_op() == "||":
                conditions.append([f"{self.prop.as_code(obj)}"])
            else:
                conditions[-1].append(f"{self.prop.as_code(obj)}")

        if len(conditions) == 1:
            return " && ".join(conditions[0])
        else:
            conds = ") || (".join(" && ".join(conds) for conds in conditions)
            return f"({conds})"


class HardAssert(Assert):
    def __init__(self, tree: Tree):
        if tree[-1].label() == "HASSERT":
            hassert = HardAssert(tree[-1])
            self.md = hassert.md
            self.objs = hassert.objs + [(Object(tree[0]), CC(tree[1]))]
            self.prop = hassert.prop
        else:
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
        if self.labels == ("OBJ", "IN", "OBJ", "RSEP", "OBJ"):
            ident = Object(self.tree[0])
            start = Object(self.tree[2])
            end = Object(self.tree[4])

            return ident, start, end
        if self.labels == ("IN", "OBJ", "RSEP", "OBJ"):
            start = Object(self.tree[1])
            end = Object(self.tree[3])
            return None, start, end
        # TODO: What other cases other than "up to" (eg. up to and not including)
        if self.labels == ("IN", "IN", "OBJ"):
            end = Object(self.tree[2])
            return None, None, end

        # if self.labels == ("IN", "IN", "OBJ"):
        #     ident = Object(self.tree[0])
        #     start = Object(self.tree[2])
        #     end = Object(self.tree[4])
        #
        #     return ident, start, end

        raise ValueError(f"Range case not handled: {self.labels}")


class UpperBound(Enum):
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


class RangeMod(Range):
    def __init__(self, tree: Tree):
        super().__init__(tree[0])
        if tree[-1].label() == "JJ":
            self.upper_bound = UpperBound.from_str(tree[-1]).as_code_lt()
        else:
            self.upper_bound = UpperBound.EXCLUSIVE.as_code_lt()


class CC:
    def __init__(self, tree: Tree):
        self.tree = tree

    def bool_op(self):
        return {
            "or": "||",
            "and": "&&",
        }[self.tree[0].lower()]


class ForEach:
    def __init__(self, tree: Tree):
        if tree[0].label() == "FOREACH":
            # another range condition here
            sub_foreach = ForEach(tree[0])
            self.tree = sub_foreach.tree
            self.quant = sub_foreach.quant
            self.range = sub_foreach.range
            self.labels = sub_foreach.labels
            if tree[1].label() == "CC":
                self.range_conditions = sub_foreach.range_conditions + [(CC(tree[1]), ModRelation(tree[-1]))]
            else:
                self.range_conditions = sub_foreach.range_conditions + [(ModRelation(tree[-1]))]

        else:
            self.tree = tree
            self.quant = Quantifier(tree[0])
            self.range = RangeMod(tree[1]) if len(tree) > 1 else None
            self.labels = tuple(x.label() for x in tree)
            self.range_conditions = []

    def as_code(self, cond: str):
        return self.with_conditions(cond)

    def with_conditions(self, post_cond: Union[List[str], str], pre_cond: Union[List[str], str] = None, flip=False):
        if not self.quant.is_universal:
            raise UnsupportedSpec("Prusti does not support existential quantifiers.")
        ## (a && b) || (c && d && e) || (f)
        # need type hint about second
        # type hints: start.as_code(), end.as_code()
        #                                    v do type introspection here
        # detect multiple identifiers.
        if isinstance(post_cond, list):
            post_cond = " && ".join(post_cond)
        if isinstance(pre_cond, list):
            pre_cond = " && ".join(pre_cond)

        xtype = "int"
        xtype_min = {
            "int": Literal(["-1"])
        }
        if self.range is not None:
            ident, start, end = self.range.as_foreach_pred()
            ident = ident or self.quant.obj
            if start is None:
                start = xtype_min[xtype]
            conditions = [
                [
                    f"{start.as_code()} <= {ident.as_code()}",
                    f"{ident.as_code()} {self.range.upper_bound} {end.as_code()}"
                ]
            ]
            if pre_cond:
                if isinstance(pre_cond, list):
                    conditions[-1].extend(pre_cond)
                else:
                    conditions[-1].append(pre_cond)
            for range_condition in self.range_conditions:
                # Has nothing attached
                if isinstance(range_condition, tuple) and range_condition[0].bool_op() == "||":
                    conditions.append([range_condition[1].as_code(ident)])
                else:
                    conditions[-1].append(range_condition.as_code(ident))

            conds = ") || (".join(" && ".join(conds) for conds in conditions)
            if flip:
                conds, post_cond = post_cond, conds
            return f"forall(|{ident.as_code()}: {xtype}| ({conds}) ==> ({post_cond}))"
        else:
            ident = self.quant.obj.as_code()
            if pre_cond:
                if flip:
                    return f"forall(|{ident}: {xtype}| {post_cond} ==> {pre_cond})"
                return f"forall(|{ident}: {xtype}| {pre_cond} ==> {post_cond})"
            return f"forall(|{ident}: {xtype}| {post_cond})"
        # raise ValueError(f"ForEach case not handled: {self.labels}")


class Quantifier:
    def __init__(self, tree: Tree):
        self.obj = Object(tree[2])
        self.coverage = tree[1]
        self.is_universal = tree[1][0].lower() in {"all", "each", "any"}


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


class QuantAssert:
    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels == ("FOREACH", "COMMA", "HASSERT"):
            self.foreach = ForEach(tree[0])
            self.assertion = HardAssert(tree[2])
        elif labels == ("FOREACH", "HASSERT"):
            self.foreach = ForEach(tree[0])
            self.assertion = HardAssert(tree[1])
        elif labels == ("HASSERT", "FOREACH"):
            self.foreach = ForEach(tree[1])
            self.assertion = HardAssert(tree[2])
        elif labels == ("CODE", "FOREACH"):
            self.foreach = ForEach(tree[1])
            self.assertion = Code(tree[0])
        else:
            raise ValueError(f"Unexpected productions for QASSERT: {labels}")

    def bound_vars(self):
        # TODO: Fix bound vars to find actual bound variables
        return self.foreach.quant.obj.as_code()

    def as_code(self):
        # eg. there exists some x such that y is true
        #     to forall: !forall(x st y is not true)
        return self.foreach.as_code(self.assertion.as_code())

    def with_conditions(self, post_cond: Union[List[str], str], pre_cond: Union[List[str], str] = None, flip=False):
        def merge_to_list(a: Union[List[str], str], b: Union[List[str], str]):
            if a is None:
                return b
            if b is None:
                return a
            if isinstance(a, list):
                if isinstance(b, list):
                    return a + b
                return a + [b]
            if isinstance(b, list):
                return b + [a]
            return [a, b]

        if flip:
            return self.foreach.with_conditions(post_cond, merge_to_list(pre_cond, self.assertion.as_code()), flip)
        return self.foreach.with_conditions(merge_to_list(self.assertion.as_code(), post_cond), pre_cond, flip)

    def as_spec(self):
        if isinstance(self.assertion, HardAssert) and self.assertion.is_precondition():
            cond = "requires"
        else:
            # TODO: Assumes that code is post-condition? How to fix?
            cond = "ensures"
        return f"#[{cond}({self.as_code()})]"


class QuantPredicate:
    def __init__(self, tree: Tree):
        if tree[0].label() not in {"IFF", "IN"}:
            raise ValueError(f"Bad tree - expected IFF or IN, got {tree[0].label()}")
        if tree[1].label() != "QASSERT":
            raise ValueError(f"Bad tree - expected ASSERT, got {tree[1].label()}")

        self.iff = tree[0].label() == "IFF"
        self.assertion = QuantAssert(tree[1])

    def bound_vars(self):
        return self.assertion.bound_vars()

    def as_code(self):
        return self.assertion.as_code()

    def with_conditions(self, post_cond: Union[List[str], str], pre_cond: Union[List[str], str] = None, flip=False):
        return self.assertion.with_conditions(post_cond, pre_cond, flip)


class ReturnIf:
    def __init__(self, tree: Union[Tree, List[Tree]]):
        if tree[0].label() != "RET":
            raise ValueError(f"Bad tree - expected RET, got {tree[0].label()}")

        if tree[1].label() != "OBJ":
            raise ValueError(f"Bad tree - expected OBJ, got {tree[1].label()}")

        if tree[2].label() not in {"QPRED", "PRED"}:
            raise ValueError(f"Bad tree - expected QPRED or PRED, got {tree[2].label()}")

        # Need to find out what the ret value is.

        self.ret_val = Object(tree[1])
        if tree[2].label() == "PRED":
            self.pred = Predicate(tree[2])
        else:
            self.pred = QuantPredicate(tree[2])

    def __str__(self):
        return f"return {self.ret_val.as_code()};"

    def as_spec(self):
        ret_val = self.ret_val.as_code()
        if ret_val == "true":
            ret_assert = "result"
        elif ret_val == "false":
            ret_assert = "!result"
        else:
            ret_assert = f"(result == {ret_val})"

        if isinstance(self.pred, QuantPredicate):
            if self.pred.iff:
                # need to be able to put cond inside quant pred
                # Need to handle ret val inside QASSERT.
                return f"""#[ensures({self.pred.with_conditions(ret_assert)})]
#[ensures({self.pred.with_conditions(ret_assert, flip=True)})]"""

            return f"""#[ensures({self.pred.with_conditions(ret_assert)})]"""

        if self.pred.iff:
            return f"""#[ensures({self.pred.as_code()} ==> {ret_assert})]
#[ensures(({ret_assert}) ==> {self.pred.as_code()})]"""
        return f"""#[ensures({self.pred.as_code()} ==> {ret_assert})]"""


class IfReturn(ReturnIf):
    def __init__(self, tree: Tree):
        # Swizzle the values as needed.
        if tree[1].label() == "COMMA":
            super().__init__([tree[2], tree[3], tree[0]])
        else:
            super().__init__([tree[1], tree[2], tree[0]])


class Specification:
    def __init__(self, tree: Tree):
        self.spec = {
            "RETIF": ReturnIf,
            "IFRET": IfReturn,
            "HASSERT": HardAssert,
            "QASSERT": QuantAssert,
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
        self.rd_parser = nltk.ChartParser(self.grammar)

    def parse_sentence(self, sentence: str, idents=None):
        sentence = sentence \
            .replace("isn't", "is not") \
            .rstrip(".")

        sentence = Sentence(sentence, use_tokenizer=self.tokenizer)
        self.root_tagger.predict(sentence)

        token_dict = sentence.to_dict(tag_type="pos")
        correct_tokens(token_dict, key_words=idents)

        labels = [entity["labels"][0] for entity in token_dict["entities"]]
        words = [entity["text"] for entity in token_dict["entities"]]
        print(labels)
        print(words)

        nltk_sent = [label.value for label in labels]
        for tree in self.rd_parser.parse(nltk_sent):
            yield attach_words_to_nodes(tree, words)


class UnsupportedSpec(ValueError):
    pass


def main():
    test_cases = [
        ("Returns `true` if `self` is 0u32", "#[ensures(self == 0u32 ==> result)]")
    ]
    predicates = [
        # "I am red",
        # "It is red",
        # "The index is less than `self.len()`",
        # "The index is 0u32",
        # Well handled.
        # "Returns `true` if and only if `self` is 0u32",
        # "Returns `true` if `self` is 0u32",
        # "Returns true if self is 0u32",
        # # Need to find a corresponding function or method for this case.
        # "Returns `true` if and only if `self` is blue",
        # # Well handled.
        # "Returns `true` if the index is greater than or equal to `self.len()`",
        # "Returns `true` if the index is equal to `self.len()`",
        # "Returns `true` if the index is not equal to `self.len()`",
        # "Returns `true` if the index isn't equal to `self.len()`.",
        # "Returns `true` if the index is not less than `self.len()`",
        # "Returns `true` if the index is smaller than or equal to `self.len()`",
        # "If the index is smaller than or equal to `self.len()`, returns true.",
        # "`index` must be less than `self.len()`",
        # "for each index `i` from `0` to `self.len()`, `i` must be less than `self.len()`",
        # "for each index `i` up to `self.len()`, `i` must be less than `self.len()`",
        # "for each index `i` from `0` to `self.len()` inclusive, `i` must be less than `self.len()`",
        # "`i` must be less than `self.len()`",
        # "for each index `i` from `0` to `self.len()` inclusive, `i` must be less than or equal to `self.len()`",
        # "for each index `i` from `0` to `self.len()`, `self.lookup(i)` must not be equal to 0",
        # # differentiate btw will and must wrt result
        # "for each index `i` from `0` to `self.len()`, `self.lookup(i)` will not be the same as `result.lookup(i)`",
        # "For all indices `i` from 0 to 32, `result.lookup(i)` will be equal to `self.lookup(31 - i)`",
        # "For all indices `i` between 0 and 32, and less than `amt`, `result.lookup(i)` will be false.",
        # # "For all indices `i` between 0 and 32, or less than `amt`, `result.lookup(i)` will be false.",
        # "`self.lookup(index)` will be equal to `val`",
        # "For all `i` between 0 and 32, and not equal to `index`, `self.lookup(i)` will be unchanged.",
        # "For all `i` between 0 and 32, and not equal to `index`, `self.lookup(i)` will not change.",
        # "Returns true if self is not blue",
        # "`self` must be blue",
        # "`other.v` must not be equal to 0",
        # "For all `i` between 0 and 32, and not equal to `index`, `self.lookup(i)` will remain static.",
        "Returns `true` if and only if `self == 2^k` for all `k`.",
        "Returns `true` if and only if `self == 2^k` for any `k`."
        "For each index from 0 to 5, `self.lookup(index)` must be true.",
        "For each index up to 5, `self.lookup(index)` must be true.",
        "`a` must be between 0 and `self.len()`."
    ]
    #         "Returns `true` if and only if `self == 2^k` for some `k`."
    #                                                  ^ not as it appears in Rust (programmer error, obviously)
    #                                                   (eg. allow mathematical notation?)
    #
    # TODO: Find examples that are not supported by Prusti
    #  async fns (eg. eventually) ? (out of scope)
    #  for all / for each
    #  Greater than
    #  https://doc.rust-lang.org/std/primitive.u32.html#method.checked_next_power_of_two
    #  If the next power of two is greater than the type’s maximum value
    #  No direct support for existential
    #  For any index that meets some condition, x is true. (eg. forall)
    parser = Parser()
    for sentence in predicates:
        print("=" * 80)
        print("Sentence:", sentence)
        print("=" * 80)
        for tree in parser.parse_sentence(sentence):
            tree: nltk.tree.Tree = tree
            try:
                print(Specification(tree).as_spec())
            except LookupError as e:
                print(f"No specification found: {e}")
            except UnsupportedSpec as s:
                print(f"Specification element not supported ({s})")
            print(tree)
            print()

    for sentence, expected in test_cases:
        try:
            tree = next(parser.parse_sentence(sentence))
            assert Specification(tree).as_spec() == expected
        except AssertionError as a:
            print(a)


if __name__ == '__main__':
    main()
