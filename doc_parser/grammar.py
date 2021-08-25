import logging
from enum import auto, Enum
from typing import Optional, List, Union, Collection

from nltk import Tree
from nltk.corpus.reader import ADJ, VERB

from pyrs_ast.lib import Method, Fn

from fn_calls import InvokeToken, Rule, InvocationFactory
from lemmatizer import lemmatize

LOGGER = logging.getLogger(__name__)


class UnsupportedSpec(ValueError):
    pass


class Code:
    def __init__(self, tree: Tree, *args):
        self.code: str = tree[0]

    def as_code(self):
        return self.code.strip("`")

    def is_ident(self, idents: Collection[str]):
        return self.as_code() in idents


class Literal:
    def __init__(self, tree: Tree, *args):
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
    def from_str(cls, s: str) -> Optional["Ops"]:
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
    def __init__(self, tree, invoke_factory):
        self.lhs = Object(tree[0], invoke_factory)
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
        self.rhs = Object(tree[2], invoke_factory)

    def as_code(self):
        return f"({self.lhs.as_code()} {self.op} {self.rhs.as_code()})"


class Object:
    def __init__(self, tree: Tree, invoke_factory):
        labels = tuple(x.label() for x in tree)
        if labels not in {
            ("PRP",), ("DT", "MNN"), ("CODE",), ("LIT",), ("DT", "LIT"), ("MNN",), ("OBJ", "OP", "OBJ"),
            ("DT", "MNN", "IN", "OBJ"), ("DT", "MJJ", "IN", "OBJ"), ("FNCALL",)
        }:
            raise ValueError(f"Bad tree - expected OBJ productions, got {labels}")
        self.labels = labels
        self.tree = tree
        self.invoke_factory = invoke_factory

    def as_code(self) -> str:
        if self.labels == ("CODE",):
            return Code(self.tree[0]).as_code()
        if self.labels[-1] == "LIT":
            return Literal(self.tree[-1]).as_code()
        if self.labels == ("DT", "MNN"):
            # TODO: This assumes that val in "The val" is a variable.
            return lemmatize(self.tree[1][0][0])
        if self.labels == ("MNN",):
            # TODO: This assumes that val in "The val" is a variable.
            return lemmatize(self.tree[0][0][0])
        if self.labels == ("OBJ", "OP", "OBJ"):
            return Op(self.tree, self.invoke_factory).as_code()
        if self.labels == ("DT", "MNN", "IN", "OBJ"):
            return f"{Object(self.tree[-1], self.invoke_factory).as_code()}.{lemmatize(self.tree[1][0][0])}()"
        if self.labels == ("DT", "MJJ", "IN", "OBJ"):
            return f"{Object(self.tree[-1], self.invoke_factory).as_code()}.{lemmatize(self.tree[1][0][0])}()"
        if self.labels == ("FNCALL",):
            return self.invoke_factory(self.tree[0]).as_code()
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
    def __init__(self, tree: Tree, invoke_factory):
        if tree[0].label() == "REL":
            rel = Relation(tree[0], invoke_factory)
            self.objs = rel.objs + [(CC(tree[1]), Object(tree[2], invoke_factory))]
            self.op = rel.op
            self.negate = rel.negate
            return

        labels = tuple(x.label() for x in tree)
        if labels not in {("TJJ", "IN", "OBJ"), ("TJJ", "EQTO", "OBJ"), ("IN", "OBJ")}:
            raise ValueError(f"Bad tree - expected REL productions, got {labels}")

        self.op = Comparator.from_str(TJJ(tree[0]).get_word())
        if labels[1] == "EQTO":
            self.op = self.op.apply_eq()
        self.objs = [(None, Object(tree[-1], invoke_factory))]
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
    def __init__(self, tree: Tree, invoke_factory):
        if tree[0].label() == "RB":
            super().__init__(tree[1], invoke_factory)
            self.apply_adverb(tree[0][0])
        else:
            super().__init__(tree[0], invoke_factory)


class MVB:
    def __init__(self, tree: Tree):
        self.mod = tree[0][0] if tree[0].label() == "RB" else None
        self.word = tree[-1][0]

    def is_negation(self):
        return self.mod is not None and self.mod == "not"

    def is_change(self):
        return self.lemma() in {"modify", "alter", "change"}

    def lemma(self):
        return lemmatize(self.word.lower(), VERB)


class MJJ:
    def __init__(self, tree: Tree):
        self.mod = tree[0][0] if tree[0].label() == "RB" else None
        self.word = tree[-1][0]

    # TODO: Expand tests here.
    def is_negation(self):
        return self.mod is not None and self.mod == "not"

    # TODO: Expand tests for checking if a particular value is unchanged.
    def is_change(self):
        return self.lemma() in {"modified", "altered", "changed"}

    def is_unchanged(self):
        return self.lemma() in {"unmodified", "unaltered", "unchanged"}

    def lemma(self):
        return lemmatize(self.word.lower(), ADJ)


class Property:
    PRODUCTIONS = {
        ("MVB", "MJJ"),
        ("MVB", "MREL"),
        ("MVB", "OBJ"),
        ("MVB", "RANGEMOD"),
        ("MVB",)
    }

    def __init__(self, tree: Tree, invoke_factory):
        labels = tuple(x.label() for x in tree)
        if labels not in self.PRODUCTIONS:
            raise ValueError(f"Bad tree - expected PROP productions, got {labels}")
        self.labels = labels
        self.tree = tree
        self.mvb = MVB(self.tree[0])
        self.negate = self.mvb.is_negation()
        self.invoke_factory = invoke_factory

    # F F -> F no neg
    # F T -> T is not changed
    # T F -> T not is  changed -> is changed
    # T T -> F (if 1, neg is true, otherwise true)
    def as_code(self, lhs: Object):
        if self.labels[-1] == "MJJ":
            mjj = MJJ(self.tree[-1])
            neg = mjj.is_negation() != self.negate
            if mjj.is_change():
                sym = "!" if neg else "="
                return f"{lhs.as_code()} {sym}= old({lhs.as_code()})"
            sym = "!" if neg else ""
            return f"{sym}{lhs.as_code()}.{mjj.word}()"

        if self.labels[-1] == "MREL":
            return ModRelation(self.tree[-1], self.invoke_factory).apply_negate(self.negate).as_code(lhs)

        if self.labels[-1] == "OBJ":
            if self.mvb.word.lower() not in {"is"}:
                # TODO: Support generic properties
                raise UnsupportedSpec(f"Unexpected verb in PROPERTY ({self.mvb.word})")

            cmp = Comparator.EQ.negate(self.negate)
            return f"{lhs.as_code()} {cmp} {Object(self.tree[1], self.invoke_factory).as_code()}"

        if self.labels[-1] == "MVB":
            if self.mvb.is_change():
                sym = "=" if self.negate else "!"
                return f"{lhs.as_code()} {sym}= old({lhs.as_code()})"
            else:
                raise ValueError(f"PROP: unexpected verb in MVB case ({self.mvb.word})")

        if self.labels[-1] == "RANGEMOD":
            r = RangeMod(self.tree[-1], self.invoke_factory)
            ident, start, stop = r.as_foreach_pred()
            if ident:
                raise ValueError(f"PROP: Unexpected ident in RANGE case: {ident}")
            return f"{start.as_code()} <= {lhs.as_code()} && {lhs.as_code()} {r.upper_bound} {stop.as_code()}"

        raise ValueError(f"Case {self.labels} not handled.")


class Assert:
    def __init__(self, tree: Tree, invoke_factory):
        if tree[0].label() != "OBJ":
            raise ValueError(f"Bad tree - expected OBJ, got {tree[0].label()}")

        if tree[-1].label() == "ASSERT":
            if tree[1].label() != "CC":
                raise ValueError(f"Bad tree - expected PROP, got {tree[1].label()}")

            assertx = Assert(tree[-1], invoke_factory)
            self.objs = assertx.objs + [(Object(tree[0], invoke_factory), CC(tree[1]))]
            self.prop = assertx.prop
        else:
            if tree[1].label() != "PROP":
                raise ValueError(f"Bad tree - expected PROP, got {tree[1].label()}")

            self.objs = [(Object(tree[0], invoke_factory), None)]
            self.prop = Property(tree[1], invoke_factory)

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
    def __init__(self, tree: Tree, invoke_factory: InvocationFactory):
        if tree[-1].label() == "HASSERT":
            hassert = HardAssert(tree[-1], invoke_factory)
            self.md = hassert.md
            self.objs = hassert.objs + [(Object(tree[0], invoke_factory), CC(tree[1]))]
            self.prop = hassert.prop
        else:
            super().__init__([tree[0], tree[2]], invoke_factory)
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
    def __init__(self, tree: Tree, invoke_factory):
        self.tree = tree
        self.labels = tuple(x.label() for x in tree)
        self.invoke_factory = invoke_factory

    def as_foreach_pred(self):
        if self.labels == ("OBJ", "IN", "OBJ", "RSEP", "OBJ"):
            ident = Object(self.tree[0], self.invoke_factory)
            start = Object(self.tree[2], self.invoke_factory)
            end = Object(self.tree[4], self.invoke_factory)

            return ident, start, end
        if self.labels == ("IN", "OBJ", "RSEP", "OBJ"):
            start = Object(self.tree[1], self.invoke_factory)
            end = Object(self.tree[3], self.invoke_factory)
            return None, start, end
        # TODO: What other cases other than "up to" (eg. up to and not including)
        if self.labels == ("IN", "IN", "OBJ"):
            end = Object(self.tree[2], self.invoke_factory)
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
    def __init__(self, tree: Tree, invoke_factory):
        super().__init__(tree[0], invoke_factory)
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
    def __init__(self, tree: Tree, invoke_factory):
        if tree[0].label() == "FOREACH":
            # another range condition here
            sub_foreach = ForEach(tree[0])
            self.tree = sub_foreach.tree
            self.quant = sub_foreach.quant
            self.range = sub_foreach.range
            self.labels = sub_foreach.labels
            if tree[1].label() == "CC":
                self.range_conditions = sub_foreach.range_conditions + [
                    (CC(tree[1]), ModRelation(tree[-1], invoke_factory))
                ]
            else:
                self.range_conditions = sub_foreach.range_conditions + [(ModRelation(tree[-1], invoke_factory))]

        else:
            self.tree = tree
            self.quant = Quantifier(tree[0], invoke_factory)
            self.range = RangeMod(tree[1], invoke_factory) if len(tree) > 1 else None
            self.labels = tuple(x.label() for x in tree)
            self.range_conditions = []

    def as_code(self, cond: str):
        return self.with_conditions(cond)

    def with_conditions_iff(self, post_cond: Union[List[str], str], pre_cond: Union[List[str], str] = None, flip=False):
        if self.quant.is_universal:
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

    def with_conditions(self, post_cond: Union[List[str], str], pre_cond: Union[List[str], str] = None, flip=False):
        quant = "forall"
        if not self.quant.is_universal:
            quant = "forsome"
            # raise UnsupportedSpec("Prusti does not support existential quantifiers.")
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
            return f"{quant}(|{ident.as_code()}: {xtype}| ({conds}) ==> ({post_cond}))"
        else:
            ident = self.quant.obj.as_code()
            if pre_cond:
                if flip:
                    return f"{quant}(|{ident}: {xtype}| {post_cond} ==> {pre_cond})"
                return f"{quant}(|{ident}: {xtype}| {pre_cond} ==> {post_cond})"
            return f"{quant}(|{ident}: {xtype}| {post_cond})"
        # raise ValueError(f"ForEach case not handled: {self.labels}")


class Quantifier:
    def __init__(self, tree: Tree, invoke_factory):
        self.obj = Object(tree[2], invoke_factory)
        self.coverage = tree[1]
        self.is_universal = tree[1][0].lower() in {"all", "each", "any"}


class Predicate:
    def __init__(self, tree: Tree, invoke_factory):
        if tree[0].label() not in {"IFF", "IF"}:
            raise ValueError(f"Bad tree - expected IFF or IF, got {tree[0].label()}")
        if tree[1].label() not in {"CODE", "ASSERT", "FNCALL"}:
            raise ValueError(f"Bad tree - expected ASSERT or CODE or FNCALL, got {tree[1].label()}")

        self.iff = tree[0].label() == "IFF"
        if tree[1].label() == "CODE":
            self.assertion = Code(tree[1])
        elif tree[1].label() == "ASSERT":
            self.assertion = Assert(tree[1], invoke_factory)
        else:
            self.assertion = invoke_factory(tree[1])

    def as_code(self):
        return self.assertion.as_code()


class QuantAssert:
    def __init__(self, tree: Tree, invoke_factory):
        labels = tuple(x.label() for x in tree)
        if labels == ("FOREACH", "COMMA", "HASSERT"):
            self.foreach = ForEach(tree[0], invoke_factory)
            self.assertion = HardAssert(tree[2], invoke_factory)
        elif labels == ("FOREACH", "HASSERT"):
            self.foreach = ForEach(tree[0], invoke_factory)
            self.assertion = HardAssert(tree[1], invoke_factory)
        elif labels == ("HASSERT", "FOREACH"):
            self.foreach = ForEach(tree[1], invoke_factory)
            self.assertion = HardAssert(tree[2], invoke_factory)
        elif labels == ("CODE", "FOREACH"):
            self.foreach = ForEach(tree[1], invoke_factory)
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
    def __init__(self, tree: Tree, invoke_factory):
        if tree[0].label() not in {"IFF", "IF"}:
            raise ValueError(f"Bad tree - expected IFF or IF, got {tree[0].label()}")
        if tree[1].label() != "QASSERT":
            raise ValueError(f"Bad tree - expected ASSERT, got {tree[1].label()}")

        self.iff = tree[0].label() == "IFF"
        self.assertion = QuantAssert(tree[1], invoke_factory)

    def bound_vars(self):
        return self.assertion.bound_vars()

    def as_code(self):
        return self.assertion.as_code()

    def with_conditions(self, post_cond: Union[List[str], str], pre_cond: Union[List[str], str] = None, flip=False):
        return self.assertion.with_conditions(post_cond, pre_cond, flip)


class MReturn:
    def __init__(self, tree: Tree, invoke_factory):
        if tree.label() != "MRET":
            raise ValueError(f"Bad tree - expected MRET, got {tree.label()}")

        labels = tuple(x.label() for x in tree)
        if labels == ("RET", "OBJ"):
            self.ret_val = Object(tree[1], invoke_factory)
        elif labels == ("OBJ", "VBZ", "RET"):
            if tree[1][0].lower() != "is":
                raise ValueError(f"Expected 'is' for OBJ VBZ RET tree, got {tree[1][0]}")
            self.ret_val = Object(tree[0], invoke_factory)
        elif labels == ("OBJ", "RET"):
            self.ret_val = Object(tree[0], invoke_factory)
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
    def __init__(self, tree: Union[Tree, List[Tree]], invoke_factory: InvocationFactory):
        if tree[0].label() == "RETIF":
            lhs = ReturnIf(tree[0], invoke_factory)
            self.preds = lhs.preds
            self.ret_vals = lhs.ret_vals
            if tree[1][0].lower() != "otherwise":
                raise ValueError(f"Unexpected RB {tree[1][0]} in ReturnIf")
            if tree[2].label() == "RETIF":
                rhs = ReturnIf(tree[2], invoke_factory)
                self.preds += rhs.preds
                self.ret_vals += rhs.ret_vals
            else:
                self.ret_vals.append(Object(tree[2], invoke_factory))
                self.preds.append(Negate(self.preds[-1]))
            return
        if tree[0].label() == "MRET":
            super().__init__(tree[0], invoke_factory)
            pred = tree[1]
        elif tree[-1].label() == "MRET":
            super().__init__(tree[-1], invoke_factory)
            pred = tree[0]
        else:
            raise ValueError("Did not find MRET")

        self.ret_vals = [self.ret_val]
        if pred.label() == "PRED":
            self.preds = [Predicate(pred, invoke_factory)]
        elif pred.label() == "QPRED":
            self.preds = [QuantPredicate(pred, invoke_factory)]
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
    def __init__(self, tree: Tree, invoke_factory):
        super().__init__(tree[0], invoke_factory)
        if tree[1][0].lower() != "otherwise":
            raise ValueError(f"Unexpected RB {tree[1][0]} in IfElseRet")
        self.op_ret_val = Object(tree[2], invoke_factory)

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
    def __init__(self, tree: Tree, invoke_factory):
        self.target = Object(tree[0], invoke_factory)
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
                self.target = Object(tree[0], invoke_factory)
                self.inputs = [Object(tree[target_i + 1], invoke_factory)]
            elif tree[target_i][0].lower() in {"to"}:
                self.target = Object(tree[target_i + 1], invoke_factory)
                self.inputs = [Object(tree[0], invoke_factory)]
        else:
            self.fn = MVB(tree[2])
            self.fn_mod = None
            self.inputs = None

    def as_spec(self) -> str:
        # x is applied to b
        # a is affected by b
        # a is stored in b

        vb = self.fn.lemma()
        op = Ops.from_str(self.fn.word.lower())
        if op is not None:
            if op == Ops.NEGATE:
                return f"#[ensures(*{self.target.as_code()} == !*{self.target.as_code()})]"
            if op == Ops.SHIFT:
                if self.fn_mod is None:
                    raise ValueError("Side Effect: got shift operation without corresponding direction.")
                op = op.apply_dir(self.fn_mod.word)
            return f"#[ensures(*{self.target.as_code()} == old(*{self.target.as_code()}) {op} {self.inputs[0].as_code()})]"
        else:
            raise UnsupportedSpec(f"Not expecting non-op side effects (got {vb})")


class Specification:
    def __init__(self, tree: Tree, invoke_factory: "InvocationFactory"):
        self.spec = {
            "RETIF": ReturnIf,
            "RETIFELSE": ReturnIfElse,
            "HASSERT": HardAssert,
            "QASSERT": QuantAssert,
            "MRET": MReturn,
            "SIDE": SideEffect,
            "FNCALL": invoke_factory,
        }[tree[0].label()](tree[0], invoke_factory)

    def as_spec(self):
        return self.spec.as_spec()


def generate_constructor_from_grammar(fn: Fn, grammar: List[InvokeToken]):
    grammar_str = " ".join(str(x) for x in grammar)
    LOGGER.info(f"Creating constructor for fn {fn.ident}: {grammar_str}")

    class CustomFnCall:
        def __init__(self, tree: Tree, invoke_factory):
            self.det = None
            self.fn = fn
            self.inputs = {ty.ident: None for ty in self.fn.inputs}
            mystery_inputs = []
            self.is_method = isinstance(fn, Method)

            for subtree, itoken in zip(tree, grammar):
                # Connectives
                if not isinstance(itoken.word, Rule):
                    continue
                if itoken.symbol() == "MD":
                    self.det = subtree[0]
                else:
                    obj = {
                        "CODE": Code,
                        "OBJ": Object,
                        "LIT": Literal,
                        "IDENT": Literal,
                    }[itoken.symbol()](subtree, invoke_factory)

                    if itoken.word.ident:
                        self.inputs[itoken.word.ident] = obj
                    else:
                        mystery_inputs.append(obj)

            input_iter = iter(self.inputs.items())
            for mystery_input in mystery_inputs:
                key, val = next(input_iter)
                while val:
                    key, val = next(input_iter)
                self.inputs[key] = mystery_input

        def as_code(self):
            inputs = iter(self.inputs.values())
            if self.is_method:
                xself = next(inputs).as_code()
                fn_ident = f"{xself}.{self.fn.ident}"
            else:
                fn_ident = self.fn.ident

            inputs = ", ".join(x.as_code() for x in inputs)
            return f"{fn_ident}({inputs})"

        def as_spec(self):
            if self.det:
                if self.det in {"will"}:
                    return f"#[ensures({self.as_code()})]"
                elif self.det in {"must"}:
                    return f"#[requires({self.as_code()})]"
            raise UnsupportedSpec(f"Can not forward specification for {self.fn.sig_str()}.")

    return CustomFnCall