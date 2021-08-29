import json
from enum import Enum
from typing import List, Optional

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


class ExprMethod:
    """The ExprMethodCall struct from https://docs.rs/syn/1.0.75/syn/struct.ExprMethodCall.html
    """
    __slots__ = ["receiver", "method", "turbofish", "args"]

    def __init__(self, receiver: "Expr", method: str, turbofish: Optional[str], args: List["Expr"]):
        self.receiver = receiver
        self.method = method
        self.turbofish = turbofish
        self.args = args

    @classmethod
    def from_kwargs(cls, **kwargs):
        receiver = Expr(**kwargs["receiver"])
        method = kwargs["method"]
        turbofish = kwargs.get("turbofish")
        if turbofish:
            raise NotImplementedError("Turbofish currently not supported")
            # Need to handle types and const values in tb
            # tb_args = ', '.join(str(Expr(**item)) for item in turbofish)
            # turbofish = f"::<{tb_args}>"
        args = [Expr(**item) for item in kwargs["args"]]
        return cls(receiver, method, turbofish, args)

    def __str__(self):
        if isinstance(self.receiver.expr, (ExprLit, ExprPath)):
            receiver = str(self.receiver)
        else:
            receiver = f"({self.receiver})"
        args = ", ".join(str(arg) for arg in self.args)
        return f"{receiver}.{self.method}{self.turbofish or ''}({args})"


class ExprBinary:
    """The ExprBinary struct from https://docs.rs/syn/1.0.75/syn/struct.ExprBinary.html
    """

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


class ExprLit:
    """The ExprLit struct from https://docs.rs/syn/1.0.75/syn/struct.ExprLit.html
    """

    def __init__(self, **kwargs):
        lit_type, val = next(iter(kwargs.items()))
        if lit_type in {"int", "float", "str"}:
            self.type = lit_type
            self.val = val
        else:
            raise ValueError(f"Unexpected literal ({lit_type}: {val})")

    def __str__(self):
        return self.val


class ExprStruct:
    """The ExprStruct struct from https://docs.rs/syn/1.0.75/syn/struct.ExprStruct.html
    """

    def __init__(self, **kwargs):
        pass

    def __str__(self):
        return "STRUCT"


class ExprPath:
    """The ExprPath struct from https://docs.rs/syn/1.0.75/syn/struct.ExprPath.html
    """

    def __init__(self, **kwargs):
        self.path = Path(**kwargs)

    def __str__(self):
        return str(self.path)


class Expr:
    """The Expr enum from https://docs.rs/syn/1.0.75/syn/enum.Expr.html
    """

    DISPATCH = {
        "lit": ExprLit,
        "struct": ExprStruct,
        "binary": ExprBinary.from_kwargs,
        "path": ExprPath,
        "method_call": ExprMethod.from_kwargs,
    }

    def __init__(self, **kwargs):
        expr_type, val = next(iter(kwargs.items()))
        constructor = self.DISPATCH.get(expr_type)
        if constructor:
            self.expr = constructor(**val)
        else:
            raise ValueError(f"{expr_type}: {val}")

    @classmethod
    def from_str(cls, s: str):
        return cls(**json.loads(astx.parse_expr(s)))

    def __str__(self):
        return str(self.expr)

