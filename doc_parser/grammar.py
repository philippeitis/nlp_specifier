import logging
from copy import copy
from enum import auto, Enum
from typing import Optional, List, Union, Collection

from nltk import Tree
from nltk.corpus.reader import ADJ, VERB

from pyrs_ast.expr import BinOp, ExprBinary
from pyrs_ast.expr import UnaryOp

from pyrs_ast.lib import Method, Fn

from fn_calls import InvokeToken, Rule, InvocationFactory
from lemmatizer import lemmatize

LOGGER = logging.getLogger(__name__)


class UnsupportedSpec(ValueError):
    pass


class Code:
    def __init__(self, tree: Tree, *_args):
        self.code: str = tree[0]

    def as_code(self):
        return self.code.strip("`")

    def is_ident(self, idents: Collection[str]):
        return self.as_code() in idents


class Literal:
    def __init__(self, tree: Union[Tree, List[str]], *_args):
        self.str: str = tree[0]

    def as_code(self):
        return self.str


def binop_from_str(s: str) -> Optional[BinOp]:
    return {
        "add": BinOp.Add,
        "added": BinOp.Add,
        "increment": BinOp.Add,
        "incremented": BinOp.Add,
        "plus": BinOp.Add,
        "sub": BinOp.Sub,
        "decremented": BinOp.Sub,
        "decrement": BinOp.Sub,
        "subtract": BinOp.Sub,
        "subtracted": BinOp.Sub,
        "div": BinOp.Div,
        "divide": BinOp.Div,
        "divided": BinOp.Div,
        "mul": BinOp.Mul,
        "multiply": BinOp.Mul,
        "multiplied": BinOp.Mul,
        "rem": BinOp.Rem,
        "remainder": BinOp.Rem,
        "xor": BinOp.BitXor,
        "and": BinOp.And,
        "or": BinOp.Or,
    }.get(s.lower())


def apply_jj(bin_op: BinOp, jj: str) -> BinOp:
    jj = jj.lower()
    if bin_op == bin_op.And:
        if jj == "logical":
            return bin_op.And
        if jj in {"boolean", "bitwise"}:
            return bin_op.BitAnd
    if bin_op in bin_op.Or:
        if jj == "logical":
            return bin_op.Or
        if jj in {"boolean", "bitwise"}:
            return bin_op.BitOr
    return bin_op


def shift_with_dir(dir_: str) -> BinOp:
    dir_ = dir_.lower()
    if dir_ == "right":
        return BinOp.Shr
    if dir_ == "left":
        return BinOp.Shl
    raise ValueError(f"Invalid direction for shift op ({dir_})")


def unaryop_from_str(s: str) -> Optional[UnaryOp]:
    return {
        "negate": UnaryOp.Neg,
        "negated": UnaryOp.Neg,
    }.get(s.lower())


class Op(ExprBinary):
    @classmethod
    def from_tree(cls, tree, invoke_factory):
        lhs = Object(tree[0], invoke_factory)
        rhs = Object(tree[2], invoke_factory)
        op_tree = tree[1][0]
        if op_tree.label() == "BITOP":
            op = apply_jj(binop_from_str(op_tree[1][0]), op_tree[0][0])
        elif op_tree.label() == "ARITHOP":
            op = binop_from_str(op_tree[0][0])
            # Include division? Not ambiguous, but ???
            if op in {BinOp.Sub} and len(op_tree) == 2 and op_tree[1][0].lower() in {"from"}:
                lhs, rhs = rhs, lhs
        elif op_tree.label() == "SHIFTOP":
            if op_tree[0].label() == "SHIFT":
                op = shift_with_dir(op_tree[3][0])
            else:
                op = shift_with_dir(op_tree[0][0])
        else:
            raise ValueError(f"Unexpected op: {op_tree}")
        return cls(lhs, op, rhs)

    def as_code(self):
        return f"({self.left.as_code()} {self.op} {self.right.as_code()})"


class PropertyOf:
    def __init__(self, tree: Tree, invoke_factory):
        if tree[1].label() == "MNN":
            self.prop = lemmatize(tree[1][0][0])
        elif tree[1].label() == "MJJ":
            self.prop = MJJ(tree[1]).lemma()
        else:
            raise ValueError("Unexpected production in PropOf")

    def as_code(self, obj):
        if self.prop == "remainder" and isinstance(obj.obj, Op):
            if obj.obj.op == BinOp.Div:
                op = copy(obj.obj)
                op.op = BinOp.Rem

                return op.as_code()
        return f"{obj.as_code()}.{self.prop}()"


class Object:
    DISPATCH = {
        ("CODE",): lambda t, inv: Code(t[0]),
        ("LIT",): lambda t, inv: Literal(t[-1]),
        ("DT", "LIT"): lambda t, inv: Literal(t[-1]),
        ("DT", "MNN"): lambda t, inv: lemmatize(t[1][0][0]),
        ("DT", "VBG", "MNN"): lambda t, inv: f"{lemmatize(t[2][0][0])}.{lemmatize(t[1][0], VERB)}()",
        ("MNN",): lambda t, inv: lemmatize(t[0][0][0]),
        ("OBJ", "OP", "OBJ"): lambda t, inv: Op.from_tree(t, inv),
        ("FNCALL",): lambda t, inv: inv(t),
        ("PROP_OF", "OBJ"): lambda t, inv: PropertyOf(t[0], inv).as_code(Object(t[1], inv))
    }

    def __init__(self, tree: Tree, invoke_factory):
        labels = tuple(x.label() for x in tree)
        fn = self.DISPATCH.get(labels)

        if fn:
            self.obj = fn(tree, invoke_factory)
        elif labels == ("PRP",):
            raise ValueError("Object: PRP case not handled.")
        else:
            raise ValueError(f"Bad tree - expected OBJ productions, got {labels}")

    def as_code(self) -> str:
        if hasattr(self.obj, "as_code") and callable(self.obj.as_code):
            return self.obj.as_code()
        return self.obj

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
    __slots__ = ["objs", "op", "negate"]

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

        if labels == ("IN", "OBJ"):
            # TODO: This is very much incorrect.
            self.op = Comparator.from_str("less")
        else:
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
            if self.mvb.word.lower() not in {"is", "be"}:
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
            if r.ident:
                raise ValueError(f"PROP: Unexpected ident in RANGE case: {r.ident}")
            return f"{r.start.as_code()} <= {lhs.as_code()} && {lhs.as_code()} {r.upper_bound} {r.end.as_code()}"

        raise ValueError(f"Case {self.labels} not handled.")


class Assert:
    def __init__(self, tree: Union[list, Tree], invoke_factory):
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
        labels = tuple(x.label() for x in tree)
        if labels == ("OBJ", "IN", "OBJ", "RSEP", "OBJ"):
            self.ident = Object(tree[0], invoke_factory)
            self.start = Object(tree[2], invoke_factory)
            self.end = Object(tree[4], invoke_factory)
        elif labels == ("IN", "OBJ", "RSEP", "OBJ"):
            self.ident = None
            self.start = Object(tree[1], invoke_factory)
            self.end = Object(tree[3], invoke_factory)
        # TODO: What other cases other than "up to" (eg. up to and not including)
        elif labels == ("IN", "IN", "OBJ"):
            self.ident = None
            self.start = None
            self.end = Object(tree[2], invoke_factory)
        else:
            raise ValueError(f"Range case not handled: {labels}")


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
            self.upper_bound = UpperBound.from_str(tree[-1][0]).as_code_lt()
        else:
            self.upper_bound = UpperBound.EXCLUSIVE.as_code_lt()


class CC:
    def __init__(self, tree: Tree):
        self.cc = tree[0].lower()

    def bool_op(self):
        return {
            "or": "||",
            "and": "&&",
        }[self.cc]


class ForEach:
    def __init__(self, tree: Tree, invoke_factory):
        if tree[0].label() == "FOREACH":
            # another range condition here
            sub_foreach = ForEach(tree[0], invoke_factory)
            self.quant = sub_foreach.quant
            self.range = sub_foreach.range
            if tree[1].label() == "CC":
                self.range_conditions = sub_foreach.range_conditions + [
                    (CC(tree[1]), ModRelation(tree[-1], invoke_factory))
                ]
            else:
                self.range_conditions = sub_foreach.range_conditions + [(None, ModRelation(tree[-1], invoke_factory))]

        else:
            self.quant = Quantifier(tree[0], invoke_factory)
            self.range = RangeMod(tree[1], invoke_factory) if len(tree) > 1 else None
            self.range_conditions = []

    def as_code(self, cond: str):
        return self.with_conditions(cond)

    def with_conditions_iff(self, post_cond: Union[List[str], str], pre_cond: Union[List[str], str] = None, flip=False):
        raise NotImplementedError()
        # if self.quant.is_universal:
        #     raise UnsupportedSpec("Prusti does not support existential quantifiers.")
        # # (a && b) || (c && d && e) || (f)
        # # need type hint about second
        # # type hints: start.as_code(), end.as_code()
        # #                                    v do type introspection here
        # # detect multiple identifiers.
        # if isinstance(post_cond, list):
        #     post_cond = " && ".join(post_cond)
        # if isinstance(pre_cond, list):
        #     pre_cond = " && ".join(pre_cond)

    def with_conditions(self, post_cond: Union[List[str], str], pre_cond: Union[List[str], str] = None, flip=False):
        quant = "forall"
        if not self.quant.is_universal:
            quant = "forsome"
            # raise UnsupportedSpec("Prusti does not support existential quantifiers.")
        # (a && b) || (c && d && e) || (f)
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
            "int": Literal(["0"])
        }
        if self.range is not None:
            ident = self.range.ident or self.quant.obj
            start = self.range.start or xtype_min[xtype]
            if start is None:
                start = xtype_min[xtype]
            conditions = [
                [
                    f"{start.as_code()} <= {ident.as_code()}",
                    f"{ident.as_code()} {self.range.upper_bound} {self.range.end.as_code()}"
                ]
            ]
            if pre_cond:
                if isinstance(pre_cond, list):
                    conditions[-1].extend(pre_cond)
                else:
                    conditions[-1].append(pre_cond)
            for cc, range_condition in self.range_conditions:
                # Has nothing attached
                if cc is not None and cc.bool_op() == "||":
                    conditions.append([range_condition.as_code(ident)])
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


class Quantifier:
    def __init__(self, tree: Tree, invoke_factory):
        self.obj = Object(tree[2], invoke_factory)
        self.coverage = tree[1]
        self.is_universal = tree[1][0].lower() in {"all", "each", "any"}


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


class BoolCond:
    def __init__(self, tree: Tree, invoke_factory):
        self.iff = tree[0].label() == "IFF"
        dispatch = {
            "ASSERT": Assert,
            "QASSERT": QuantAssert,
            "CODE": Code,
        }
        expr = tree[1][0]
        if expr.label() in dispatch:
            self.expr = dispatch[expr.label()](expr, invoke_factory)
        else:
            raise ValueError(f"Bad tree - expected PRED, QPRED or CODE, got {expr.label()}")

    def as_code(self):
        return self.expr.as_code()

    def negated(self):
        return Negated(self.expr, self.iff)


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


class Negated:
    def __init__(self, expr, iff):
        if isinstance(expr, (Assert, Code)):
            self.iff = iff
            self.expr = expr
        elif isinstance(expr, QuantAssert):
            raise ValueError("Quantifiers can not be negated")
        else:
            raise ValueError(f"Unexpected type {type(expr)} can not be negated")

    def as_code(self):
        return f"!{self.expr.as_code()}"


class ReturnIf(MReturn):
    __slots__ = ["preds", "ret_vals"]

    def __init__(self, tree: Union[Tree, List[Tree]], invoke_factory: InvocationFactory):
        if tree[0].label() == "RETIF":
            lhs = ReturnIf(tree[0], invoke_factory)
            self.preds = lhs.preds
            self.ret_vals = lhs.ret_vals
            if tree[2][0].lower() != "otherwise":
                raise ValueError(f"Unexpected RB {tree[1][0]} in ReturnIf")
            if tree[3].label() == "RETIF":
                rhs = ReturnIf(tree[3], invoke_factory)
                self.preds += rhs.preds
                self.ret_vals += rhs.ret_vals
            else:
                self.ret_vals.append(Object(tree[3], invoke_factory))
                self.preds.append(self.preds[-1].negated())
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
        self.preds = [BoolCond(pred, invoke_factory)]

    def __str__(self):
        return f"return {self.ret_val.as_code()};"

    @classmethod
    def spec(cls, pred: Union[BoolCond, Negated], ret_val: Object) -> str:
        ret_val = ret_val.as_code()
        if ret_val == "true":
            ret_assert = "result"
        elif ret_val == "false":
            ret_assert = "!result"
        else:
            ret_assert = f"(result == {ret_val})"

        if isinstance(pred.expr, QuantAssert):
            expr = pred.expr
            if pred.iff:
                # need to be able to put cond inside quant pred
                # Need to handle ret val inside QASSERT.
                return f"""#[ensures({expr.with_conditions(ret_assert)})]
#[ensures({expr.with_conditions(ret_assert, flip=True)})]"""

            return f"""#[ensures({expr.with_conditions(ret_assert)})]"""

        if pred.iff:
            return f"""#[ensures({pred.as_code()} ==> {ret_assert})]
#[ensures(({ret_assert}) ==> {pred.as_code()})]"""
        return f"""#[ensures({pred.as_code()} ==> {ret_assert})]"""

    def as_spec(self):
        return "\n".join(self.spec(p, rv) for p, rv in zip(self.preds, self.ret_vals))


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
        word = self.fn.word.lower()
        op = binop_from_str(word) or unaryop_from_str(word)
        if op is None:
            if word in {"shift", "shifted"}:
                if self.fn_mod:
                    op = shift_with_dir(self.fn_mod.word)
                elif self.fn.mod:
                    op = shift_with_dir(self.fn.mod)
                else:
                    raise ValueError("Side Effect: got shift operation without corresponding direction.")

        if op is not None:
            if op == UnaryOp.Neg:
                return f"#[ensures(*{self.target.as_code()} == !*old({self.target.as_code()}))]"
            return f"#[ensures(*{self.target.as_code()} == old(*{self.target.as_code()}) {op} {self.inputs[0].as_code()})]"
        else:
            raise UnsupportedSpec(f"Not expecting non-op side effects (got {vb})")


class Specification:
    def __init__(self, tree: Tree, invoke_factory: "InvocationFactory"):
        self.spec = {
            "RETIF": ReturnIf,
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
