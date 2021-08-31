from typing import Optional

from . import expr


class Binding:
    def __init__(self, **kwargs):
        self.ident = kwargs["ident"]
        self.ty = Type(**kwargs["ty"])

    def __str__(self):
        return f"{self.ident} = {self.ty}"


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
                arg_type, arg_val = next(iter(arg.items()))
                if arg_type == "type":
                    res.append(str(Type(**arg_val)))
                elif arg_type == "lifetime":
                    res.append(str(LifetimeParam(**arg)))
                elif arg_type == "binding":
                    res.append(str(Binding(**arg_val)))
                else:
                    raise ValueError(f"{arg_type}, {arg}")
            return f"{self.ident}<{', '.join(res)}>"
        elif arg_type == "parenthesized":
            inputs = [Type(**t) for t in args["inputs"]]
            output = "" if args["output"] is None else f" -> {Type(**args['output'])}"
            return f"{self.ident}({', '.join(str(input) for input in inputs)}){output}"

        raise ValueError("Unexpected argument", self.arguments)


class Path:
    def __init__(self, **kwargs):
        self.segments = [Segment(**segment) for segment in kwargs["segments"]]

    def __getitem__(self, i: int):
        return self.segments[i]

    def __str__(self):
        return "::".join([str(segment) for segment in self.segments])

    def ident(self):
        return self.segments[-1].ident


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
        self.mut = kwargs.get("mut", False)
        self.lifetime = kwargs.get("lifetime")

    def __str__(self):
        lifetime = f"'{self.lifetime} " if self.lifetime else ""
        mut = f"mut " if self.mut else ""
        return f"&{lifetime}{mut}{self.elem}"

    def name(self):
        if isinstance(self.elem, SingletonType):
            return self.elem.path


class PtrType:
    def __init__(self, **kwargs):
        self.elem = Type(**kwargs["elem"])
        self.mut = kwargs.get("mut", False)
        self.const = kwargs.get("const", False)
        assert not (self.mut and self.const), "can not be both const and mut"

    def __str__(self):
        if self.const:
            return f"*const {self.elem}"
        else:
            return f"*mut {self.elem}"

    def name(self):
        if isinstance(self.elem, SingletonType):
            return self.elem.path


class SliceType:
    def __init__(self, **kwargs):
        self.elem = Type(**kwargs["elem"])

    def __str__(self):
        return f"[{self.elem}]"

    def name(self):
        if isinstance(self.elem, SingletonType):
            return self.elem.path


class TraitObjectType:
    def __init__(self, **kwargs):
        self.dyn = kwargs.get("dyn", False)
        self.bounds = IdentParam(**kwargs)

    def __str__(self):
        return f"{'dyn ' if self.dyn else ''}{self.bounds.bound_str()}"


class TraitType:
    def __init__(self, **kwargs):
        self.modifier = "modifier" in kwargs and kwargs["modifier"] == "maybe"
        self.path = Path(**kwargs["path"])

    def __str__(self):
        return f"{'?' if self.modifier else ''}{self.path}]"


class ArrayType:
    def __init__(self, **kwargs):
        self.elem = Type(**kwargs["elem"])
        self.len = expr.Expr(**kwargs["len"])

    def __str__(self):
        return f"[{self.elem}; {self.len}]"


TYPE_DICT = {
    "path": SingletonType,
    "tuple": TupleType,
    "reference": RefType,
    "array": ArrayType,
    "ptr": PtrType,
    "trait_object": TraitObjectType,
    "slice": SliceType,
    "trait": TraitType,
    "bare_fn": lambda **kwargs: print("bare_fn", kwargs),
}


class EmptyType:
    def __str__(self):
        return "()"


class NeverType:
    def __str__(self):
        return "!"


class Type:
    def __init__(self, **kwargs):
        try:
            type_key, val = next(iter(kwargs.items()))
            self.ty = TYPE_DICT[type_key](**val)

            self.methods = []
        except StopIteration:
            self.ty = EmptyType()
            self.methods = []
        self.struct = None

    def register_struct(self, struct):
        # Don't clobber self.struct with duplicate methods.
        if not self.struct:
            self.struct = struct
            struct.methods += self.methods
            self.methods = struct.methods

    def __str__(self):
        return str(self.ty)

    def register_method(self, method: "Method"):
        self.methods.append(method)

    def name(self) -> Optional[Path]:
        if isinstance(self.ty, SingletonType):
            return self.ty.path
        if isinstance(self.ty, RefType):
            return self.ty.name()


class LifetimeParam:
    def __init__(self, **kwargs):
        self.lifetime = kwargs.get("lifetime")

    def __str__(self):
        return f"'{self.lifetime}"


class IdentParam:
    def __init__(self, **kwargs):
        self.ident = kwargs.get("ident")
        self.bounds = []
        for bound in kwargs.get("bounds", []):
            key, val = next(iter(bound.items()))
            if key == "trait":
                self.bounds.append(Path(**val["path"]))
            elif key == "lifetime":
                self.bounds.append(LifetimeParam(**bound))

    def bound_str(self):
        return " + ".join(str(bound) for bound in self.bounds)

    def __str__(self):
        if self.bounds:
            return f"{self.ident}: {self.bound_str()}"
        return self.ident


class TypeParam:
    def __init__(self, **kwargs):
        if "lifetime" in kwargs:
            self.param = LifetimeParam(**kwargs["lifetime"])
        elif "type" in kwargs:
            self.param = IdentParam(**kwargs["type"])
        else:
            raise ValueError(f"Expected lifetime or type in kwargs, got {kwargs}")

    def __str__(self):
        return str(self.param)
