from enum import Enum, auto
from typing import Optional, List, Union

try:
    from .docs import Docs
    from .ast_types import Path, Type, TypeParam
    from .scope import Scope
except ImportError as e:
    from docs import Docs
    from ast_types import Path, Type, TypeParam
    from scope import Scope

import astx
import json


def ast_items_from_json(scope, items: [], parent_type=None) -> []:
    results = []
    for i, item in enumerate(items):
        item: dict = item
        item_kind = next(iter(item.keys()))
        results.append(KEY_TO_CLASS[item_kind](scope=scope, parent_type=parent_type, **item[item_kind]))
    return results


def indent(block):
    return "\n".join(["    " + s for s in str(block).splitlines()])


class LitExpr:
    def __init__(self, **kwargs):
        pass


class StructExpr:
    def __init__(self, **kwargs):
        pass


class Expr:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        expr_type, val = next(iter(self.kwargs.items()))
        if expr_type == "lit":
            lit_type, val = next(iter(val.items()))
            if lit_type == "int":
                return val
            if lit_type == "str":
                return val
            raise ValueError(f"{expr_type}, {lit_type}: {val}")
        if expr_type == "struct":
            return "STRUCT"
        raise ValueError(f"{expr_type}: {val}")


class Attr:
    def __init__(self, **kwargs):
        self.ident = Path(**kwargs["path"])
        self.style = kwargs["style"]
        self.tokens = TokenStream(kwargs.get("tokens", []))

    def __str__(self):
        c = "!" if self.style == "inner" else ""
        return f"#{c}[{self.ident}{self.tokens}]"

    def is_doc(self):
        return str(self.ident) == "doc"

    def doc_string(self):
        if str(self.ident) == "doc":
            return self.tokens[1].val.strip("\" ")


class LitAttr:
    def __init__(self, lit: str):
        self.lit = lit

    def __str__(self):
        return self.lit


class HasAttrs:
    def __init__(self, **kwargs):
        self.attrs: List[Union[Attr, LitAttr]] = [Attr(**attr) for attr in kwargs.get("attrs", [])]
        self.docs = self._extract_docs()

    def _extract_docs(self) -> Docs:
        docs = Docs()
        for attr in self.attrs:
            doc = attr.doc_string()
            if doc:
                docs.push_line(doc)
        return docs

    def fmt_attrs(self) -> str:
        if self.attrs:
            return "\n".join([str(attr) for attr in self.attrs]) + "\n"
        return ""


class HasParams:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = [TypeParam(**param) for param in kwargs.get("generics", {}).get("params", [])]
        self.where_clause = kwargs.get("where_clause")

    def fmt_generics(self) -> str:
        if self.params:
            return f"<{', '.join([str(x) for x in self.params])}>"
        return ""

    def is_generic(self) -> bool:
        return len(self.params) != 0


class HasItems:
    def __init__(self, scope=None, parent_type=None, **kwargs):
        super().__init__(**kwargs)
        self.items = ast_items_from_json(scope or Scope(), kwargs.get("items", []), parent_type=parent_type)


class TokenType(Enum):
    PUNCT = auto()
    LIT = auto()
    GROUP = auto()
    IDENT = auto()

    @classmethod
    def from_str(cls, s: str):
        return {
            "punct": cls.PUNCT,
            "lit": cls.LIT,
            "group": cls.GROUP,
            "ident": cls.IDENT,
        }[s]


class Delimiter(Enum):
    PARENTHESIS = auto()
    BRACE = auto()
    BRACKET = auto()
    NONE = auto()

    @classmethod
    def from_str(cls, s: str):
        return {
            "parenthesis": cls.PARENTHESIS,
            "brace": cls.BRACE,
            "bracket": cls.BRACKET,
            "none": cls.NONE,
        }[s]


class Token:
    def __init__(self, **kwargs):
        assert len(kwargs) == 1
        key, val = next(iter(kwargs.items()))
        self.key = TokenType.from_str(key)
        self.val = val

    def __str__(self):
        if self.key == TokenType.PUNCT:
            return self.val["op"]  # ("" if self.val["spacing"] == "joint" else " ")
        if self.key == TokenType.LIT:
            return self.val
        if self.key == TokenType.GROUP:
            sym = {
                Delimiter.PARENTHESIS: ("(", ")"),
                Delimiter.BRACKET: ("[", "]"),
                Delimiter.BRACE: ("{", "}")
            }[Delimiter.from_str(self.val["delimiter"])]
            return f"{sym[0]} {TokenStream(self.val['stream'])} {sym[1]}]"

        if self.key == TokenType.IDENT:
            return self.val
        return f"[{self.key}, {self.val}]"

    def is_ident(self):
        return self.key == TokenType.IDENT

    def is_dot_punct(self):
        return self.key == TokenType.PUNCT and self.val["op"] == "."

    def is_joint_punct(self):
        return self.key == TokenType.PUNCT and self.val["spacing"] == "joint"


class TokenStream:
    def __init__(self, tokens):
        self.tokens = [Token(**token) for token in tokens]

    def __str__(self):
        if len(self.tokens) == 0:
            return ""

        if len(self.tokens) == 1:
            return str(self.tokens[0])

        last_token = self.tokens[0]
        s = str(last_token)
        for token in self.tokens[1:]:
            if last_token.is_ident() and token.is_ident():
                s += f" {token}"
            elif last_token.is_dot_punct() and token.is_ident():
                s += str(token)
            elif last_token.is_joint_punct():
                s += str(token)
            elif last_token.is_ident() and token.key == TokenType.PUNCT:
                s += str(token)
            elif last_token.key == TokenType.PUNCT and token.is_ident():
                s += str(token)
            else:
                s += " " + str(token)
            last_token_key = token
        return s

    def __getitem__(self, item):
        return self.tokens[item]


class Receiver:
    def __init__(self, **kwargs):
        self.mut = kwargs.get("mut", False)
        self.ref = kwargs.get("ref", False)
        self.lifetime = kwargs.get("lifetime")
        self.ty = None

    def __str__(self):
        lifetime = "" if self.lifetime is None else f"'{self.lifetime} "
        return f"{'&' if self.ref else ''}{lifetime}{'mut ' if self.mut else ''}self"


class Fn(HasParams, HasAttrs):
    # TODO: Provide context as to the type of self.
    def __init__(self, scope=None, **kwargs):
        super().__init__(**kwargs)
        self.ident = kwargs["ident"]
        self.inputs = kwargs["inputs"]
        if kwargs["output"] is None:
            self.output = scope.define_type()
        else:
            self.output = scope.define_type(**kwargs["output"])
        for i, inputx in enumerate(self.inputs):
            if "receiver" in inputx:
                self.inputs[i] = Receiver(**inputx["receiver"])
            else:
                self.inputs[i] = BoundVariable(scope=scope, **inputx["typed"])
        scope.add_fn(self.ident, self)

    def type_tuples(self) -> [(str, str)]:
        types = []
        for inputx in self.inputs:
            if isinstance(inputx, Receiver):
                types.append(("?", "self"))
            elif isinstance(inputx, BoundVariable):
                types.append((str(inputx.ty), inputx.ident))
            else:
                raise ValueError("Unexpected type in self.inputs")
        return types

    def __str__(self):
        return f"{self.fmt_attrs()}{self.sig_str()}"

    def sig_str(self):
        inputs = ", ".join([str(x) for x in self.inputs])
        return f"fn {self.ident}{self.fmt_generics()}({inputs}) -> {self.output};"


class BoundVariable:
    def __init__(self, scope=None, **kwargs):
        self.ty = scope.define_type(**kwargs["ty"])
        self.ident = kwargs["pat"]["ident"]["ident"]

    def __str__(self):
        return f"{self.ident}: {self.ty}"


class NamedField(HasAttrs):
    def __init__(self, scope=None, **kwargs):
        super().__init__(**kwargs)
        self.ty = scope.define_type(**kwargs["ty"])
        self.ident = kwargs["ident"]

    def __str__(self):
        return f"{self.fmt_attrs()}{self.ident}: {self.ty}"


class Const(HasAttrs):
    def __init__(self, scope=None, **kwargs):
        super().__init__(**kwargs)
        self.ident = kwargs["ident"]
        self.ty = scope.define_type(**kwargs["ty"])
        self.expr = Expr(**kwargs["expr"])

    def __str__(self):
        return f"{self.fmt_attrs()}const {self.ident}: {self.ty} = {self.expr};"


class Fields:
    def __init__(self, scope, kwargs):
        if kwargs == "unit":
            self.is_unit = True
            return
        self.is_unit = False
        type_key, fields = next(iter(kwargs.items()))
        fn = {
            "named": lambda **x: NamedField(scope=scope, **x),
            "unnamed": lambda **x: scope.define_type(**x["ty"])
        }[type_key]
        self.style = type_key
        self.fields = [fn(**field) for field in fields]

    def named(self):
        return self.style == "named"

    def __iter__(self):
        return iter(self.fields)


class Struct(HasParams, HasAttrs):
    def __init__(self, scope=None, **kwargs):
        super().__init__(**kwargs)
        self.ident = kwargs["ident"]
        self.fields = Fields(scope, kwargs["fields"])
        self.methods = []
        scope.add_struct(self.ident, self)

    def __str__(self):
        if self.fields.is_unit:
            fields = ";"
        elif self.fields.named():
            fields = " {\n" + ",\n".join([indent(field) for field in self.fields]) + ",\n}"
        else:
            fields = "(" + ", ".join([indent(field) for field in self.fields]) + ");"
        return f"{self.fmt_attrs()}struct {self.ident}{self.fmt_generics()}{fields}"

    def name(self):
        return self.ident


class Method(Fn):
    def __init__(self, parent_type=None, **kwargs):
        super().__init__(**kwargs)

        if len(self.inputs) > 0 and isinstance(self.inputs[0], Receiver):
            self.inputs[0].ty = parent_type


class Impl(HasParams, HasItems, HasAttrs):
    def __init__(self, scope=None, parent_type=None, **kwargs):
        self_path = Path(**kwargs["self_ty"]["path"])
        ty = scope.find_type(self_path.ident())
        super().__init__(scope=scope, parent_type=ty, **kwargs)
        for item in self.items:
            if isinstance(item, Fn):
                ty.methods.append(item)
        self.ty = ty
        self.concrete_path = self_path

    def __str__(self):
        items = "\n\n".join([indent(item) for item in self.items])
        return f"""{self.fmt_attrs()}impl{self.fmt_generics()} {self.concrete_path} {{
{items}
}}"""


class Mod(HasAttrs):
    def __init__(self, scope=None, **kwargs):
        super().__init__(**kwargs)
        self.ident = kwargs["ident"]
        self.content = ast_items_from_json(scope, kwargs.get("content", []))

    def __str__(self):
        content = "\n".join([indent(item) for item in self.content])
        return f"""{self.fmt_attrs()}mod {self.ident} {{
{content}
}}"""


class Use(HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = []
        next_layer = kwargs["tree"]
        self.glob = False
        while "ident" not in next_layer:
            if next_layer == "*":
                self.glob = True
                break
            next_layer = next_layer["path"]
            self.layers.append(next_layer["ident"])
            next_layer = next_layer["tree"]

        if not self.glob:
            self.layers.append(next_layer["ident"])
        else:
            self.layers.append("*")

    def __str__(self):
        return f"{self.fmt_attrs()}use {'::'.join(self.layers)};"


class EnumVariant:
    def __init__(self, scope=None, **kwargs):
        self.fields = Fields(scope, kwargs["fields"])
        self.ident = kwargs["ident"]

    def __str__(self):
        if self.fields.is_unit:
            return self.ident
        if self.fields.named():
            fields = " {\n    " + ",\n    ".join([str(field) for field in self.fields]) + ",\n}"
        else:
            fields = "(" + ", ".join([str(field) for field in self.fields]) + ")"

        return f"{self.ident}{fields}"


class Enum(HasParams, HasAttrs):
    def __init__(self, scope=None, **kwargs):
        super().__init__(**kwargs)
        self.ident = kwargs["ident"]
        self.variants = [EnumVariant(scope, **v) for v in kwargs["variants"]]

    def __str__(self):
        enum_vals = ",\n    ".join([str(v) for v in self.variants])
        return f"""{self.fmt_attrs()}enum{self.fmt_generics()} {self.ident} {{
    {enum_vals}
}}"""


class ExternCrate(HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ident = kwargs["ident"]

    def __str__(self):
        return f"{self.fmt_attrs()}extern crate {self.ident};"


KEY_TO_CLASS = {
    "fn": Fn,
    "const": Const,
    "typed": BoundVariable,
    "struct": Struct,
    "impl": Impl,
    "method": Method,
    "use": Use,
    "mod": Mod,
    "extern_crate": ExternCrate,
    "enum": Enum,
}


class LexError(ValueError):
    pass


class AstFile(HasItems, HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shebang: Optional[str] = kwargs.get("shebang")
        self.scope = kwargs["scope"]

    @classmethod
    def from_str(cls, s: str, scope=None):
        result = astx.ast_from_str(s)
        try:
            return cls(scope=scope or Scope(), **json.loads(result))
        except json.decoder.JSONDecodeError:
            pass
        raise LexError(result)

    @classmethod
    def from_path(cls, path, scope=None):
        with open(path, "r") as file:
            code = file.read()
        return cls.from_str(code, scope=scope)
