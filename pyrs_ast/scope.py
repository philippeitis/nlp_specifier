from collections import defaultdict
from typing import Dict, List, Optional, Union, Collection

from .ast_types import Type, Segment, SelfType
from .query import Query


# Acts like a type factory, ensuring that only one instance of a type exists for a particular declaration.
class Scope:
    def __init__(self):
        self.structs: Dict[str, "Struct"] = {}
        self.functions: Dict[str, "Fn"] = {}
        self.modules = defaultdict(Scope)

    def add_struct(self, name: str, struct):
        self.structs[name] = struct

    def add_fn(self, name: str, fn):
        self.functions[name] = fn

    def find_function(self, fn: Union[str, List[str]]) -> Optional["Fn"]:
        if isinstance(fn, str):
            fn = fn.split("::")
        if len(fn) == 1:
            return self.functions.get(fn[0])
        else:
            return self.modules[fn[0]].find_function(fn[1:])

    def find_type(self, ty: Union[str, List[str]]) -> Optional[Union[Type, "Struct"]]:
        if isinstance(ty, str):
            ty = ty.split("::")
        if len(ty) == 1:
            return self.structs.get(ty[0])
        else:
            return self.modules[ty[0]].find_type(ty[1:])

    def attach_struct(self, ty, path: Optional[List[Segment]] = None):
        if path is None:
            name = ty.name()
            if name is None:
                return ty
            else:
                path = name.segments

        if len(path) != 1:
            return self.modules[str(path[0])].attach_struct(ty, path[1:])

        name = path[0].ident
        if name == "Self":
            return SelfType()
        ty.register_struct(self.structs[name])

    def find_fn_matches(self, query: Query) -> Collection["Fn"]:
        """Return all functions and methods which match the particular set of queries, in no particular order."""
        res = set()
        for ty in self.structs.values():
            for method in ty.methods:
                if query.matches_item(method):
                    res.add(method)
        for fn in self.functions.values():
            if query.matches_item(fn):
                res.add(fn)

        return res

    def find_item(self, path: List[Segment]):
        if len(path) == 1:
            return self.find_type(str(path[0])) or self.find_function(str(path[0]))

        if path[0].ident in self.modules:
            return self.modules[path[0].ident].find_item(path[1:])
