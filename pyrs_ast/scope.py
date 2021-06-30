from typing import Dict

from pyrs_ast.types import Type


class Scope:
    def __init__(self):
        self.named_types = {}
        self.structs = {}
        self.functions: Dict[str, "Fn"] = {}

        self.parent = None
        self.visibility = None

    def add_struct(self, name: str, struct):
        self.structs[name] = struct

    def find_function(self, fn: str):
        return self.functions.get(fn)

    def find_type(self, ty: str):
        return self.named_types.get(ty)

    def define_type(self, **kwargs) -> Type:
        ty = Type(**kwargs)
        name = ty.name()
        if name is None:
            return ty
        name = str(name)

        if name in self.named_types:
            return self.named_types[name]

        self.named_types[name] = ty
        return ty
