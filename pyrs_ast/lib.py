from enum import Enum, auto
from typing import Optional, List, Tuple

import astx
import jsons
import json
from pprint import pprint


def ast_items_from_json(items: []) -> []:
    results = []
    for i, item in enumerate(items):
        item: dict = item
        item_kind = next(iter(item.keys()))
        results.append(KEY_TO_CLASS[item_kind](**item[item_kind]))
    return results


def indent(block):
    return "\n".join(["    " + s for s in str(block).splitlines()])


class IdentPrinter:
    def __init__(self):
        self.indentation = 0

    def indent(self):
        self.indentation += 4

    def dedent(self):
        self.indentation = max(0, self.indentation - 4)

    def print(self, s: str):
        for line in s.splitlines():
            print(" " * self.indentation + line)


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


class Segment:
    def __init__(self, **kwargs):
        self.arguments = kwargs.get("arguments")
        self.ident = kwargs["ident"]

    def __str__(self):
        if self.arguments is None:
            return self.ident
        arg_type, args = next(iter(self.arguments.items()))
        if arg_type == "angle_bracketed":
            args = args["args"]
            res = []
            for arg in args:
                arg_type, arg = next(iter(arg.items()))
                if arg_type == "type":
                    res.append(str(Type(**arg)))
                else:
                    raise ValueError(f"{arg_type}, {arg}")
            return f"{self.ident}<{', '.join(res)}>"


class Path:
    def __init__(self, **kwargs):
        self.segments = [Segment(**segment) for segment in kwargs["segments"]]

    def __str__(self):
        return "::".join([str(segment) for segment in self.segments])


class Attr:
    def __init__(self, **kwargs):
        self.ident = Path(**kwargs["path"])
        self.style = kwargs["style"]
        self.tokens = TokenStream(kwargs.get("tokens", []))

    def __str__(self):
        c = "!" if self.style == "inner" else ""
        # print(f"IDENT: {self.ident}")
        # if str(self.ident) == "ensures":
        #     print([(token, str(token)) for token in self.tokens])
        return f"#{c}[{self.ident}{self.tokens}]"

    def is_doc(self):
        return str(self.ident) == "doc"

    def doc_string(self):
        if str(self.ident) == "doc":
            return self.tokens[1].val.strip("\" ")


class HasAttrs:
    def __init__(self, **kwargs):
        self.attrs = [Attr(**attr) for attr in kwargs.get("attrs", [])]

    def extract_docs(self):
        docs = Docs()
        for attr in self.attrs:
            doc = attr.doc_string()
            if doc:
                docs.push_line(doc)
        return docs

    def fmt_attrs(self):
        if self.attrs:
            return "\n".join([str(attr) for attr in self.attrs]) + "\n"
        return ""


class SingletonType:
    def __init__(self, **kwargs):
        self.path = Path(**kwargs)

    def __str__(self):
        return str(self.path)


class TupleType:
    def __init__(self, **kwargs):
        self.elems = [Type(**elem) for elem in kwargs["elems"]]

    def __str__(self):
        elems = ", ".join([str(elem) for elem in self.elems])
        return f"({elems})"


class RefType:
    def __init__(self, **kwargs):
        self.elem = Type(**kwargs["elem"])
        self.lifetime = kwargs.get("lifetime")

    def __str__(self):
        if self.lifetime:
            return f"&'{self.lifetime} {self.elem}"
        else:
            return f"&{self.elem}"


TYPE_DICT = {
    "path": SingletonType,
    "tuple": TupleType,
    "reference": RefType
}


class EmptyType:
    def __str__(self):
        return "()"


class Type:
    def __init__(self, **kwargs):
        try:
            type_key, val = next(iter(kwargs.items()))
            self.ty = TYPE_DICT[type_key](**val)
        except StopIteration:
            self.ty = EmptyType()

    def __str__(self):
        return str(self.ty)


class TokenType(Enum):
    PUNCT = auto()
    LIT = auto()
    GROUP = auto()
    IDENT = auto()

    @classmethod
    def from_str(cls, s: str):
        if s == "punct":
            return cls.PUNCT
        if s == "lit":
            return cls.LIT
        if s == "group":
            return cls.GROUP
        if s == "ident":
            return cls.IDENT


class Delimiter(Enum):
    PARENTHESIS = auto()
    BRACE = auto()
    BRACKET = auto()
    NONE = auto()

    @classmethod
    def from_str(cls, s: str):
        if s == "parenthesis":
            return cls.PARENTHESIS
        if s == "brace":
            return cls.BRACE
        if s == "bracket":
            return cls.BRACKET
        if s == "none":
            return cls.NONE


class Token:
    def __init__(self, **kwargs):
        assert len(kwargs) == 1
        key, val = next(iter(kwargs.items()))
        self.key = TokenType.from_str(key)
        self.val = val
        # print(f"TOKEN: {kwargs} -> {self}")

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
        # print("TOKEN STREAM START")
        self.tokens = [Token(**token) for token in tokens]
        # print("TOKEN STREAM END")

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


class Docs:
    def __init__(self):
        self.sections: [Tuple[Optional[str], List[str]]] = [(None, [])]

    def push_line(self, line: str):
        line_strip = line.strip("\" ")
        if line_strip.startswith("#"):
            self.sections.append((line, []))
        else:
            self.sections[-1][1].append(line_strip)

    def section_headers(self) -> [str]:
        return [section[0] for section in self.sections[1:]]

    def is_empty(self):
        return len(self.sections) == 1 and len(self.sections[0][1]) == 0


class Receiver:
    def __init__(self, **kwargs):
        self.mut = kwargs.get("mut", False)
        self.ref = kwargs.get("ref", False)
        self.lifetime = kwargs.get("lifetime")

    def __str__(self):
        lifetime = "" if self.lifetime is None else f"'{self.lifetime} "
        return f"{'&' if self.ref else ''}{lifetime}{'mut ' if self.mut else ''}self"


class Fn(HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ident = kwargs["ident"]
        self.inputs = kwargs["inputs"]
        self.generics = kwargs.get("generics")
        if kwargs["output"] is None:
            self.output = Type()
        else:
            self.output = Type(**kwargs["output"])
        for i, inputx in enumerate(self.inputs):
            if "receiver" in inputx:
                self.inputs[i] = Receiver(**inputx["receiver"])
            else:
                self.inputs[i] = BoundVariable(**inputx["typed"])
        # self.vis = kwargs.get("attrs", [])
        # self.sig = kwargs.get("signature")
        # self.block = kwargs.get("block")

    def __str__(self):
        inputs = ", ".join([str(x) for x in self.inputs])
        return f"{self.fmt_attrs()}fn {self.ident}({inputs}) -> {self.output};"

    def is_generic(self):
        return self.generics is not None

    def extract_docs(self):
        docs = Docs()
        for attr in self.attrs:
            doc = attr.doc_string()
            if doc:
                docs.push_line(doc)
        return docs


class BoundVariable:
    def __init__(self, **kwargs):
        self.ty = Type(**kwargs["ty"])
        self.ident = kwargs["pat"]["ident"]["ident"]

    def __str__(self):
        return f"{self.ident}: {self.ty}"


class NamedField(HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ty = Type(**kwargs["ty"])
        self.ident = kwargs["ident"]

    def __str__(self):
        return f"{self.fmt_attrs()}{self.ident}: {self.ty}"


class Const(HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ident = kwargs["ident"]
        self.ty = Type(**kwargs["ty"])
        self.expr = Expr(**kwargs["expr"])

    def __str__(self):
        return f"{self.fmt_attrs()}const {self.ident}: {self.ty} = {self.expr};"


class Fields:
    def __init__(self, kwargs):
        if kwargs == "unit":
            self.is_unit = True
            return
        self.is_unit = False
        type_key, fields = next(iter(kwargs.items()))
        fn = {
            "named": NamedField,
            "unnamed": lambda **x: Type(**x["ty"])
        }[type_key]
        self.style = type_key
        self.fields = [fn(**field) for field in fields]

    def named(self):
        return self.style == "named"

    def __iter__(self):
        return iter(self.fields)


class Struct(HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generics = kwargs.get("generics")
        self.ident = kwargs["ident"]
        self.fields = Fields(kwargs["fields"])

    def __str__(self):
        if self.fields.is_unit:
            fields = ";"
        elif self.fields.named():
            fields = " {\n" + ",\n".join([indent(field) for field in self.fields]) + ",\n}"
        else:
            fields = "(" + ", ".join([indent(field) for field in self.fields]) + ");"
        return f"{self.fmt_attrs()}struct {self.ident}{fields}"

    def is_generic(self):
        return self.generics is not None


class Method(Fn):
    pass


class Impl(HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ty = Path(**kwargs["self_ty"]["path"])
        self.items = ast_items_from_json(kwargs.get("items", []))

    def __str__(self):
        items = "\n\n".join([indent(item) for item in self.items])
        return f"""{self.fmt_attrs()}impl {self.ty} {{
{items}
}}"""


class Mod(HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ident = kwargs["ident"]
        self.content = ast_items_from_json(kwargs.get("content", []))

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
    def __init__(self, **kwargs):
        self.fields = Fields(kwargs["fields"])
        self.ident = kwargs["ident"]

    def __str__(self):
        if self.fields.is_unit:
            return self.ident
        if self.fields.named():
            fields = " {\n    " + ",\n    ".join([str(field) for field in self.fields]) + ",\n}"
        else:
            fields = "(" + ", ".join([str(field) for field in self.fields]) + ")"

        return f"{self.ident}{fields}"


class Enum(HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ident = kwargs["ident"]
        self.variants = [EnumVariant(**v) for v in kwargs["variants"]]

    def __str__(self):
        enum_vals = ",\n    ".join([str(v) for v in self.variants])
        return f"""{self.fmt_attrs()}enum {self.ident} {{
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


class AstFile(HasAttrs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shebang: Optional[str] = kwargs.get("shebang")
        self.items = ast_items_from_json(kwargs.get("items", []))


class LexError(ValueError):
    pass


def read_ast_from_str(s: str) -> AstFile:
    result = astx.ast_from_str(s)
    try:
        return AstFile(**jsons.loads(result))
    except json.decoder.JSONDecodeError:
        pass
    raise LexError(result)


def read_ast_from_path(path):
    with open(path, "r") as file:
        code = file.read()
    return read_ast_from_str(code)


def print_ast_docs(ast: AstFile):
    for item in ast.items:
        if isinstance(item, HasAttrs):
            docs = item.extract_docs()
            if docs.is_empty():
                pass
            else:
                print(docs.sections)


def print_ast(ast: AstFile):
    for attr in ast.attrs:
        print(attr)

    for item in ast.items:
        print(item)
        print()


def main():
    ast = read_ast_from_path("pyrs_ast/test.rs")
    print("PRINTING DOCS")
    print_ast_docs(ast)
    print("PRINTING FILE")
    print_ast(ast)


if __name__ == '__main__':
    main()

    # Add examples
    # Motivate problems with what is being accomplished
    # problem and solution and reflection - therefore we do this
