from typing import Dict, Union, List

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
        res = self.named_types.get(ty, self.structs.get(ty))
        return res

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


class FnArg:
    def __init__(self, xtype, keyword: str = None, position: int = None):
        self.type = xtype
        self.keyword = keyword
        self.position = position


class QueryField:
    def __init__(self, item: Union[List[str], FnArg], synonyms: bool, optional: bool):
        self.item = item
        self.synonyms = synonyms
        self.optional = optional


class Query:
    def __init__(self, fields):
        self.fields = fields

    def matches(self, item):
        raise NotImplementedError()
