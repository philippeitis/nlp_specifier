from enum import Enum


class ExprArray:
    pass


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

    def __str__(self):
        if self.op.is_assign():
            return f"{self.left} {self.op.value} {self.right};"
        return f"{self.left} {self.op.value} {self.right}"


class UnaryOp(str, Enum):
    Deref = "*"
    Not = "!"
    Neg = "-"
