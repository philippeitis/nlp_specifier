import json
from enum import Enum

from astx import astx

from .ast_types import Path


class ExprArray:
    pass


class UnaryOp(str, Enum):
    Deref = "*"
    Not = "!"
    Neg = "-"


class BinOp(str, Enum):
    Add = "+"
    Sub = "-"
    Mul = "*"
    Div = "/"
    Rem = "%"
    And = "&&"
    Or = "||"
    BitXor = "^"
    BitAnd = "&"
    BitOr = "|"
    Shl = "<<"
    Shr = ">>"
    Eq = "=="
    Lt = "<"
    Le = "<="
    Ne = "!="
    Ge = ">="
    Gt = ">"
    AddEq = "+="
    SubEq = "-="
    MulEq = "*="
    DivEq = "/="
    RemEq = "%="
    BitXorEq = "^="
    BitAndEq = "&="
    BitOrEq = "|="
    ShlEq = "<<="
    ShrEq = ">>="

    def is_assign(self):
        return self in {
            BinOp.AddEq, BinOp.SubEq, BinOp.MulEq, BinOp.DivEq,
            BinOp.RemEq, BinOp.BitXorEq, BinOp.BitAndEq, BinOp.BitOrEq,
            BinOp.ShlEq, BinOp.ShrEq

        }

    def __str__(self):
        return self.value


class ExprBinary:
    __slots__ = ["left", "op", "right"]

    def __init__(self, left, op: BinOp, right):
        self.left = left
        self.op = op
        self.right = right

    @classmethod
    def from_kwargs(cls, **kwargs):
        left = Expr(**kwargs["left"])
        op = BinOp(kwargs["op"])
        right = Expr(**kwargs["right"])
        return cls(left, op, right)

    def __str__(self):
        if self.op.is_assign():
            return f"{self.left} {self.op.value} {self.right};"
        return f"{self.left} {self.op.value} {self.right}"


class LitExpr:
    def __init__(self, **kwargs):
        lit_type, val = next(iter(kwargs.items()))
        if lit_type in {"int", "float", "str"}:
            self.type = lit_type
            self.val = val
        else:
            raise ValueError(f"Unexpected literal ({lit_type}: {val})")

    def __str__(self):
        return self.val


class StructExpr:
    def __init__(self, **kwargs):
        pass

    def __str__(self):
        return "STRUCT"


class PathExpr:
    def __init__(self, **kwargs):
        self.path = Path(**kwargs)

    def __str__(self):
        return str(self.path)


class Expr:
    DISPATCH = {
        "lit": LitExpr,
        "struct": StructExpr,
        "binary": ExprBinary.from_kwargs,
        "path": PathExpr
    }

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def from_str(cls, s: str):
        return cls(**json.loads(astx.parse_expr(s)))

    def __str__(self):
        expr_type, val = next(iter(self.kwargs.items()))
        constructor = self.DISPATCH.get(expr_type)

        if constructor:
            return str(constructor(**val))
        raise ValueError(f"{expr_type}: {val}")
