from enum import Enum, auto
from typing import List, Union, Tuple, Optional
import logging

from flair.data import Sentence, Label, Token, Tokenizer
from flair.models import MultiTagger

import nltk
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()

logger = logging.getLogger("flair")
logger.setLevel(logging.ERROR)


class CodeTokenizer(Tokenizer):
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


def might_be_obj(word: str, key_words=None) -> Tuple[bool, str]:
    """Determines whether `word` is likely to be code - specifically,
     if the word is a Rust literal or a function argument.

    :param word:
    :param key_words:
    :return:
    """
    # detect bool literals
    if word.lower() in {"true", "false"}:
        return True, word.lower()

    if key_words and word in key_words:
        return True, word

    # detect numbers
    if word.endswith(("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64")):
        return True, word

    if word.isnumeric():
        return True, word

    return False, word


def apply_operation_tokens(token_dict):
    entities = token_dict["entities"]

    def get_label(t) -> str:
        return t["labels"][0].value

    def set_label(t, label: str):
        t["labels"][0] = Label(label)

    def text(t) -> str:
        return t["text"]

    def set_text(t, word: str):
        t["text"] = word

    def is_obj(t: str) -> bool:
        return get_label(t) in {
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

    for i, entity in enumerate(entities):
        if i == 0:
            continue

        if is_obj(entity):
            delta = len(entities) - i - 1
            if delta >= 6 and is_obj(entities[i + 6]):
                # SHIFT IN DT NN IN
                if LEMMATIZER.lemmatize(text(entities[i + 1]).lower(), "v") == "shift" \
                        and get_label(entities[i + 2]) == "IN" \
                        and get_label(entities[i + 3]) == "DT" \
                        and get_label(entities[i + 4]).startswith("NN") \
                        and get_label(entities[i + 5]) == "IN":
                    set_label(entities[i + 1], "SHIFT")
            if delta >= 3 and is_obj(entities[i + 3]):
                # JJ SHIFT
                if get_label(entities[i + 1]).startswith("JJ") \
                        and LEMMATIZER.lemmatize(text(entities[i + 2]).lower(), "v") == "shift":
                    set_label(entities[i + 2], "SHIFT")
                # ARITH IN
                elif is_arith(entities[i + 1]) \
                        and get_label(entities[i + 2]) == "IN":
                    set_label(entities[i + 1], "ARITH")
            # ARITH
            if delta >= 2 and is_obj(entities[i + 2]):
                if is_arith(entities[i + 1]) or text(entities[i + 1]).lower() == "xor":
                    set_label(entities[i + 1], "ARITH")


def correct_tokens(token_dict, key_words=None):
    def get_label(t) -> str:
        return t["labels"][0].value

    def set_label(t, label: Label):
        t["labels"][0] = label

    def text(t) -> str:
        return t["text"]

    def set_text(t, word: str):
        t["text"] = word

    def is_obj(t: str) -> bool:
        return t in {
            "NN", "NNP", "NNPS", "NNS", "CODE", "LIT"
        }

    def get_pos(label: str) -> str:
        if is_obj(label):
            return "n"
        if label.startswith("VB"):
            return "v"
        raise ValueError(f"Have not handled get_pos case for {label}")

    entities = token_dict["entities"]

    for entity in entities:
        if text(entity).startswith("`"):
            entity["labels"][0] = Label("CODE")

    for i, entity in enumerate(entities):
        mbc, word = might_be_obj(text(entity), key_words)
        if mbc:
            set_label(entity, Label("LIT"))
            set_text(entity, word)

    for i, entity in enumerate(entities):
        try:
            word = LEMMATIZER.lemmatize(text(entity).lower(), get_pos(get_label(entity)))
        except ValueError:
            continue
        if word == "return":
            if i + 1 < len(entities) and is_obj(get_label(entities[i + 1])):
                set_label(entity, Label("RET"))
            elif i >= 2 and is_obj(get_label(entities[i - 2])) and get_label(entities[i - 1]) in {"VBZ"}:
                set_label(entity, Label("RET"))
            elif i >= 1 and is_obj(get_label(entities[i - 1])):
                set_label(entity, Label("RET"))

    if len(entities) <= 2:
        return

    for i, entity in enumerate(entities):
        if text(entity).lower() == "for":
            if i + 1 < len(entities) and get_label(entities[i + 1]) in {"DT"}:
                set_label(entity, Label("FOR"))

    apply_operation_tokens(token_dict)


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


class Ops(Enum):
    ADD = auto()
    SUB = auto()
    DIV = auto()
    MUL = auto()
    REM = auto()
    SHIFT = auto()
    SHIFTL = auto()
    SHIFTR = auto()
    AND = auto()
    ANDL = auto()
    ANDB = auto()
    OR = auto()
    ORL = auto()
    ORB = auto()
    XOR = auto()
    NEGATE = auto()

    @classmethod
    def from_str(cls, s: str) -> Optional[str]:
        return {
            "add": cls.ADD,
            "added": cls.ADD,
            "increment": cls.ADD,
            "incremented": cls.ADD,
            "plus": cls.ADD,
            "sub": cls.SUB,
            "decremented": cls.SUB,
            "decrement": cls.SUB,
            "subtract": cls.SUB,
            "subtracted": cls.SUB,
            "div": cls.DIV,
            "divide": cls.DIV,
            "divided": cls.DIV,
            "mul": cls.MUL,
            "multiply": cls.MUL,
            "multiplied": cls.MUL,
            "rem": cls.REM,
            "remainder": cls.REM,
            "xor": cls.XOR,
            "and": cls.AND,
            "or": cls.OR,
            "shift": cls.SHIFT,
            "shifted": cls.SHIFT,
            "negate": cls.NEGATE,
            "negated": cls.NEGATE,
        }.get(s.lower())

    def apply_jj(self, jj: str):
        if self == self.AND:
            if jj.lower() == "logical":
                return self.ANDL
            if jj.lower() in {"boolean", "bitwise"}:
                return self.ANDB
        if self == self.OR:
            if jj.lower() == "logical":
                return self.ORL
            if jj.lower() in {"boolean", "bitwise"}:
                return self.ORB
        return self

    def apply_dir(self, d: str):
        if self == self.SHIFT:
            if d.lower() == "right":
                return self.SHIFTR
            if d.lower() == "left":
                return self.SHIFTL
        return self

    def __str__(self):
        return {
            self.ADD: "+",
            self.SUB: "-",
            self.DIV: "/",
            self.MUL: "*",
            self.REM: "%",
            self.SHIFTL: "<<",
            self.SHIFTR: ">>",
            self.ANDL: "&&",
            self.ANDB: "&",
            self.ORL: "||",
            self.ORB: "|",
            self.XOR: "^",
        }[self]


class Op:
    def __init__(self, tree):
        self.lhs = Object(tree[0])
        op = tree[1][0]
        if op.label() == "BITOP":
            self.op = Ops.from_str(op[1][0]).apply_jj(op[0][0])
        elif op.label() == "ARITHOP":
            self.op = Ops.from_str(op[0][0])
        elif op.label() == "SHIFTOP":
            if op[0].label() == "SHIFT":
                self.op = Ops.SHIFT.apply_dir(op[3][0])
            else:
                self.op = Ops.SHIFT.apply_dir(op[0][0])
        else:
            raise ValueError(f"Unexpected op: {op}")
        self.rhs = Object(tree[2])

    def as_code(self):
        return f"({self.lhs.as_code()} {self.op} {self.rhs.as_code()})"


class Object:
    def __init__(self, tree: Tree):
        labels = tuple(x.label() for x in tree)
        if labels not in {("PRP",), ("DT", "MNN"), ("CODE",), ("LIT",), ("DT", "LIT"), ("MNN",), ("OBJ", "OP", "OBJ")}:
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
            return LEMMATIZER.lemmatize(self.tree[1][0][0])
        if self.labels == ("MNN",):
            # TODO: This assumes that val in "The val" is a variable.
            return LEMMATIZER.lemmatize(self.tree[0][0][0])
        if self.labels == ("OBJ", "OP", "OBJ"):
            return Op(self.tree).as_code()

        raise ValueError(f"{self.labels} not handled.")

    def is_output(self):
        raise NotImplementedError()
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
            rel = Relation(tree[0])
            self.objs = rel.objs + [(CC(tree[1]), Object(tree[2]))]
            self.op = rel.op
            self.negate = rel.negate
            return

        labels = tuple(x.label() for x in tree)
        if labels not in {("TJJ", "IN", "OBJ"), ("TJJ", "EQTO", "OBJ"), ("IN", "OBJ")}:
            raise ValueError(f"Bad tree - expected REL productions, got {labels}")

        self.op = Comparator.from_str(TJJ(tree[0]).get_word())
        if labels[1] == "EQTO":
            self.op = self.op.apply_eq()
        self.objs = [(None, Object(tree[-1]))]
        self.negate = False

    def as_code(self, lhs: Object):
        relation = self.op.negate(self.negate)
        if len(self.objs) == 1:
            return f"{lhs.as_code()} {relation} {self.objs[0][1].as_code()}"

        relations = [[f"{lhs.as_code()} {relation} {self.objs[0][1].as_code()}"]]
        for cc, obj in self.objs[1:]:
            assert isinstance(cc, CC)
            if cc.bool_op() == "||":
                relations.append([f"{lhs.as_code()} {relation} {obj.as_code()}"])
            else:
                relations[-1].append(f"{lhs.as_code()} {relation} {obj.as_code()}")

        if len(relations) == 1:
            return " && ".join(relations[0])
        else:
            relations = ") || (".join(" && ".join(rels) for rels in relations)
            return f"({relations})"

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

    def vb(self) -> str:
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

        # TODO: Implement code for type detection.
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
        if tree[1].label() not in {"CODE", "ASSERT"}:
            raise ValueError(f"Bad tree - expected ASSERT or CODE, got {tree[1].label()}")

        self.iff = tree[0].label() == "IFF"
        if tree[1].label() == "CODE":
            self.assertion = Code(tree[1])
        else:
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


class MReturn:
    def __init__(self, tree: Tree):
        if tree.label() != "MRET":
            raise ValueError(f"Bad tree - expected MRET, got {tree.label()}")

        labels = tuple(x.label() for x in tree)
        if labels == ("RET", "OBJ"):
            self.ret_val = Object(tree[1])
        elif labels == ("OBJ", "VBZ", "RET"):
            if tree[1][0].lower() != "is":
                raise ValueError(f"Expected 'is' for OBJ VBZ RET tree, got {tree[1][0]}")
            self.ret_val = Object(tree[0])
        elif labels == ("OBJ", "RET"):
            self.ret_val = Object(tree[0])
        else:
            raise ValueError(f"Unexpected tree structure - got {labels}")

    def as_spec(self):
        return f"#[ensures(result == {self.ret_val.as_code()})]"


class Negate:
    def __init__(self, pred):
        if isinstance(pred, Predicate):
            self.pred = pred
            self.iff = self.pred.iff
        elif isinstance(pred, QuantPredicate):
            raise ValueError("Quantifiers can not be negated")
        else:
            raise ValueError(f"Can't negate {type(pred)}")

    def as_code(self):
        return f"!{self.pred.as_code()}"


class ReturnIf(MReturn):
    def __init__(self, tree: Union[Tree, List[Tree]]):
        if tree[0].label() == "RETIF":
            lhs = ReturnIf(tree[0])
            self.preds = lhs.preds
            self.ret_vals = lhs.ret_vals
            if tree[1][0].lower() != "otherwise":
                raise ValueError(f"Unexpected RB {tree[1][0]} in ReturnIf")
            if tree[2].label() == "RETIF":
                rhs = ReturnIf(tree[2])
                self.preds += rhs.preds
                self.ret_vals += rhs.ret_vals
            else:
                self.ret_vals.append(Object(tree[2]))
                self.preds.append(Negate(self.preds[-1]))
            return
        if tree[0].label() == "MRET":
            super().__init__(tree[0])
            pred = tree[1]
        elif tree[-1].label() == "MRET":
            super().__init__(tree[-1])
            pred = tree[0]
        else:
            raise ValueError("Did not find MRET")

        self.ret_vals = [self.ret_val]
        if pred.label() == "PRED":
            self.preds = [Predicate(pred)]
        elif pred.label() == "QPRED":
            self.preds = [QuantPredicate(pred)]
        else:
            raise ValueError(f"Bad tree - expected QPRED or PRED, got {pred.label()}")

    def __str__(self):
        return f"return {self.ret_val.as_code()};"

    @classmethod
    def spec(cls, pred: Union[Negate, Predicate, QuantPredicate], ret_val: Object) -> str:
        ret_val = ret_val.as_code()
        if ret_val == "true":
            ret_assert = "result"
        elif ret_val == "false":
            ret_assert = "!result"
        else:
            ret_assert = f"(result == {ret_val})"

        if isinstance(pred, QuantPredicate):
            if pred.iff:
                # need to be able to put cond inside quant pred
                # Need to handle ret val inside QASSERT.
                return f"""#[ensures({pred.with_conditions(ret_assert)})]
#[ensures({pred.with_conditions(ret_assert, flip=True)})]"""

            return f"""#[ensures({pred.with_conditions(ret_assert)})]"""

        if pred.iff:
            return f"""#[ensures({pred.as_code()} ==> {ret_assert})]
#[ensures(({ret_assert}) ==> {pred.as_code()})]"""
        return f"""#[ensures({pred.as_code()} ==> {ret_assert})]"""

    def as_spec(self):
        return "\n".join(self.spec(p, rv) for p, rv in zip(self.preds, self.ret_vals))


class ReturnIfElse(ReturnIf):
    def __init__(self, tree: Tree):
        super().__init__(tree[0])
        if tree[1][0].lower() != "otherwise":
            raise ValueError(f"Unexpected RB {tree[1][0]} in IfElseRet")
        self.op_ret_val = Object(tree[2])

    def as_spec(self):
        if_part = super().as_spec()
        ret_val = self.op_ret_val.as_code()
        if ret_val == "true":
            ret_assert = "result"
        elif ret_val == "false":
            ret_assert = "!result"
        else:
            ret_assert = f"(result == {ret_val})"

        if isinstance(self.pred, QuantPredicate):
            raise ValueError("Can not negate quantifier in IfRetElse.")

        if self.pred.iff:
            return f"""{if_part}\n#[ensures(!{self.pred.as_code()} ==> {ret_assert})]
        #[ensures(({ret_assert}) ==> {self.pred.as_code()})]"""
        return f"""{if_part}\n#[ensures(!{self.pred.as_code()} ==> {ret_assert})]"""


class SideEffect:
    def __init__(self, tree: Tree):
        self.target = Object(tree[0])
        if len(tree) > 3:
            if tree[2].label() == "MJJ":
                self.fn_mod = MJJ(tree[2])
                self.fn = MVB(tree[3])
                target_i = 4
            else:
                self.fn = MVB(tree[2])
                self.fn_mod = None
                target_i = 3

            if tree[target_i][0].lower() in {"by", "in"}:
                self.target = Object(tree[0])
                self.inputs = [Object(tree[target_i + 1])]
            elif tree[target_i][0].lower() in {"to"}:
                self.target = Object(tree[target_i + 1])
                self.inputs = [Object(tree[0])]
        else:
            self.fn = MVB(tree[2])
            self.fn_mod = None
            self.inputs = None

    def as_spec(self) -> str:
        # x is applied to b
        # a is affected by b
        # a is stored in b

        vb = LEMMATIZER.lemmatize(self.fn.vb().lower(), "v")
        op = Ops.from_str(self.fn.vb().lower())
        if op is not None:
            if op == Ops.NEGATE:
                return f"#[ensures(*{self.target.as_code()} == !*{self.target.as_code()})]"
            if op == Ops.SHIFT:
                if self.fn_mod is None:
                    raise ValueError("Side Effect: got shift operation without corresponding direction.")
                op = op.apply_dir(self.fn_mod.word())
            return f"#[ensures(*{self.target.as_code()} == old(*{self.target.as_code()}) {op} {self.inputs[0].as_code()})]"
        else:
            raise ValueError(f"Not expecting non-op side effects (got {vb})")


class Specification:
    def __init__(self, tree: Tree):
        self.spec = {
            "RETIF": ReturnIf,
            "RETIFELSE": ReturnIfElse,
            "HASSERT": HardAssert,
            "QASSERT": QuantAssert,
            "MRET": MReturn,
            "SIDE": SideEffect,
        }[tree[0].label()](tree[0])

    def as_spec(self):
        return self.spec.as_spec()


def get_leaf_nodes(tree: Tree, nodes: List[Tree]) -> [Tree]:
    """Returns all leaf nodes, from left to right. Modifies `nodes` in place.

    :param tree:
    :param nodes:
    :return:
    """
    if isinstance(tree, str):
        return nodes

    if len(tree) == 1 and isinstance(tree[0], str):
        nodes.append(tree)
    else:
        for sub_tree in tree:
            get_leaf_nodes(sub_tree, nodes)

    return nodes


def attach_words_to_nodes(tree: Tree, words: List[str]) -> Tree:
    """Modifies `tree` in place, by assigning each word in words to a leaf node in tree, in sequential order.

    :param tree:
    :param words:
    :return: modified tree
    """
    nodes = get_leaf_nodes(tree, [])
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

    def parse_sentence(self, sentence: str, idents=None) -> Tree:
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
        # "Returns `true` if and only if `self == 2^k` for all `k`.",
        # "Returns `true` if and only if `self == 2^k` for any `k`."
        # "For each index from 0 to 5, `self.lookup(index)` must be true.",
        # "For each index up to 5, `self.lookup(index)` must be true.",
        # "`a` must be between 0 and `self.len()`.",
        # "`a` and `b` must be equal to 0 or `self.len()`.",
        # "True is returned.",
        # "Returns `a` if `self.check(b)`, otherwise `b`",
        # "Returns `a` if `fn(a)`",
        # "If `fn(a)`, returns `a`",
        # "If `fn(a)`, returns `a`, otherwise `b`",
        # "Returns `a` logical and `b`",
        # "Returns `a` bitwise and `b`",
        # "Returns `a` xor `b`",
        # "Returns `a` multiplied by `b`",
        # "Returns `a` divided by `b`",
        # "Returns `a` subtracted by `b`",
        # "Returns `a` subtracted from `b`",
        # "Returns `a` added to `b`",
        # "Returns `a` shifted to the right by `b`",
        # "Returns `a` shifted to the left by `b`",
        # "Returns `a` left shift `b`",
        # "Returns `a` right shift `b`",
        # TODO: Side effects:
        #  Assignment operation:
        #  Assign result of fn to val
        #  Assign result of operation to val
        #   eg. Increments a by n
        #       Decrements a by n
        #       Divides a by n
        #       Increases a by n
        #       Decreases a by n
        #       Negates a
        #       Multiplies a by n
        #       Subtracts n from a
        #       Adds n to a
        #       Shifts a to the DIR by n
        #       DIR shifts a by n
        #       a is shifted to the right by n
        #       a is divided by n
        #       a is multiplied by n
        #       a is increased by n
        #       a is incremented by n
        #       a is decremented by a
        #       a is negated
        #       a is right shifted by n
        #       a is ?VB?
        # "Sets `a` to 1",
        # "Assigns 1 to `a`.",
        # "Increments `a` by 1",
        # "Adds 1 to `a`"
        "`a` is incremented by 1",
        "`a` is negated",
        "`a` is right shifted by `n`",
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
            tree: Tree = tree
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
