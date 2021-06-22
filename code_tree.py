from enum import Enum, auto
from typing import Union, Optional, List


class DuplicateNameError(ValueError):
    pass


class DuplicateMethod(ValueError):
    pass


class CodeBlock:
    pass


class Expr(CodeBlock):
    pass


class BoolExpr(Expr):
    pass


class Type:
    def __init__(self, name, values: Optional[List['BoundVariable']] = None):
        self.name = name
        self.values = values
        self.methods = {}

    def register_method(self, name, args: ['Type']):
        if name in self.methods:
            raise DuplicateMethod()
        self.methods[name] = Method(name, args)

    def get_method_call(self, name, bound_vars):
        return MethodCall(self.methods[name], bound_vars)

    def __str__(self):
        return self.name


class BoundVariable(Expr):
    def __init__(self, name, typex: Type):
        self.name = name
        self.typex = typex

    def rust_form(self):
        return f"{self.name}: {self.typex}"

    def __str__(self):
        return self.name


class Function:
    def __init__(self, name, args: [Type]):
        self.name = name
        self.args = args


class Method(Function):
    pass


class FunctionCall(Expr):
    def __init__(self, fn: Function, args: [BoundVariable], description: Optional[str] = None):
        self.description = description
        # Fetches the item at `index`
        self.fn = fn
        if not isinstance(args, list):
            self.args = [args]
        else:
            self.args = args

    def matches_description(self, phrase: str):
        # Should be used to identify which function matches the particular string.
        raise NotImplementedError()

    def __str__(self):
        args = ", ".join([str(x) for x in self.args])
        return f"{self.fn.name}({args})"


class MethodCall(FunctionCall):
    def __init__(self, fn: Method, args: [BoundVariable], description: Optional[str] = None):
        super().__init__(fn, args, description)

    def __str__(self):
        if len(self.args) == 1:
            return f"{self.args[0]}.{self.fn.name}()"
        else:
            args = ", ".join([str(x) for x in self.args[1:]])
            return f"{self.args[0]}.{self.fn.name}({args})"


class Op(Enum):
    ADD = auto()
    XOR = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    BOR = auto()
    BAND = auto()

    def __str__(self):
        return {
            Op.ADD: "+",
            Op.XOR: "^",
            Op.SUB: "-",
            Op.MUL: "*",
            Op.DIV: "/",
            Op.BOR: "|",
            Op.BAND: "&"
        }[self]


class BooleanOp(Enum):
    # Comparison
    GT = auto()
    GTE = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LTE = auto()
    AND = auto()
    OR = auto()

    # Operators

    def __str__(self):
        return {
            BooleanOp.GT: ">",
            BooleanOp.GTE: ">=",
            BooleanOp.EQ: "==",
            BooleanOp.NEQ: "!=",
            BooleanOp.LT: "<",
            BooleanOp.LTE: "<=",
            BooleanOp.AND: "&&",
            BooleanOp.OR: "||",
        }[self]


class Comparison(BoolExpr):
    def __init__(self, lhs: Union[Expr, str], rhs: Union[Expr, str], operator: Union[Op, BooleanOp]):
        self.lhs = lhs
        self.rhs = rhs
        self.operator = operator

    def __str__(self):
        return f"{self.lhs} {self.operator} {self.rhs}"


class Conditional(BoolExpr):
    def __init__(self, antecedent: Union[Expr, str], consequent: Union[Expr, str]):
        self.antecedent = antecedent
        self.consequent = consequent

    def __str__(self):
        return f"({self.antecedent}) ==> ({self.consequent})"


class BoundVariableCollection:
    def __init__(self, bound_types: [BoundVariable]):
        names = set()
        for bt in bound_types:
            if bt.name in names:
                raise DuplicateNameError(f"Name '{bt.name}' seen more than once in bound types.")
            names.add(bt.name)

        self.bound_types = bound_types

    def __iter__(self):
        return iter(self.bound_types)


class ForAll:
    def __init__(self, bound_types: BoundVariableCollection, bool_expr: Union[BoolExpr, str]):
        self.bound_types = bound_types
        self.bool_expr = bool_expr

    def __str__(self):
        bound_vars = ", ".join([bt.rust_form() for bt in self.bound_types])

        return f"forall(|{bound_vars}| {self.bool_expr})"


class Condition:
    def __init__(self, condition: Union[BoolExpr, str, ForAll]):
        self.condition = condition


class PreCondition(Condition):
    def __str__(self):
        return f"#[requires({self.condition})]"


class PostCondition(Condition):
    def __str__(self):
        return f"#[ensures({self.condition})]"


if __name__ == '__main__':
    """
    1. Scan entire codebase, fetch all function names, descriptions, types
        -> visit AST, extract function names, 
    2. Build up types and add all methods
    3. Add descriptions of names to methods
    4. Begin parsing method specifications - tokenize, replace name matches with relevant variable or method or fn,
    build up tree structure of components
    5. Verify everything
    
    """
    usize = Type("usize")
    x = BoundVariable("x", usize)
    index = BoundVariable("index", usize)
    listx = Type("List")

    listx.register_method("lookup", [listx, usize])
    listx.register_method("len", [listx])

    self = BoundVariable("self", listx)
    old = Function("old", [])

    #
    forall = PostCondition(
        ForAll(
            BoundVariableCollection([x]),
            Conditional(
                Comparison(
                    Comparison(
                        Comparison("0", x, BooleanOp.LTE),  # <- "x is greater or equal to 0"
                        Comparison(x, listx.get_method_call("len", self), BooleanOp.LT),
                        BooleanOp.AND
                    ),
                    Comparison(x, index, BooleanOp.NEQ),
                    BooleanOp.AND
                ),
                Comparison(
                    listx.get_method_call("lookup", [self, x]),
                    FunctionCall(old, listx.get_method_call("lookup", [self, x])),
                    BooleanOp.EQ
                )
            )
        )
    )

    # c = a | b;
    # Each bit in c is equal to the corresponding bits in a and b, or'ed together.
    # The function is applied to each value in `self` inplace.
    # Iterates over each value in `self`.
    #
    print(forall)

    # foreach
    # Each value in a is incremented by 1.
    # (|a: &mut u8| a += 1)

    # TODO:
    # design writeup
    # bert library
    # begin translating trees
    # search around
