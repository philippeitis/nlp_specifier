from typing import Dict, List, Optional, Union, Collection

from .ast_types import Type


class QueryField:
    def matches(self, fn: "Fn") -> bool:
        pass


class Query:
    def __init__(self, fields: List[QueryField]):
        self.fields = fields

    def matches_fn(self, fn) -> bool:
        return all(field.matches(fn) for field in self.fields)


# Acts like a type factory, ensuring that only one instance of a type exists for a particular declaration.
class Scope:
    def __init__(self):
        self.named_types: Dict[str, Type] = {}
        self.structs: Dict[str, "Struct"] = {}
        self.functions: Dict[str, "Fn"] = {}

        self.parent = None
        self.visibility = None

    def add_struct(self, name: str, struct):
        self.structs[name] = struct
        if name in self.named_types:
            self.named_types[name].register_struct(struct)

    def add_fn(self, name: str, fn):
        self.functions[name] = fn

    def find_function(self, fn: str) -> Optional["Fn"]:
        return self.functions.get(fn)

    def find_type(self, ty: str) -> Optional[Union[Type, "Struct"]]:
        return self.named_types.get(ty, self.structs.get(ty))

    def define_type(self, **kwargs) -> Type:
        ty = Type(**kwargs)
        name = ty.name()

        if name is None:
            return ty

        name = str(name)

        if name in self.named_types:
            return self.named_types[name]
        if name in self.structs:
            ty.register_struct(self.structs[name])
        self.named_types[name] = ty
        return ty

    def find_fn_matches(self, query: Query) -> Collection["Fn"]:
        """Return all functions and methods which match the particular set of queries, in no particular order."""
        res = set()
        for ty in self.structs.values():
            for method in ty.methods:
                if query.matches_fn(method):
                    res.add(method)
        for ty in self.named_types.values():
            for method in ty.methods:
                if query.matches_fn(method):
                    res.add(method)
        for fn in self.functions.values():
            if query.matches_fn(fn):
                res.add(fn)

        return res


class FnArg(QueryField):
    def __init__(self, xtype, position: int = None, is_input: bool = True):
        self.type = xtype
        self.position = position
        self.is_input = is_input

    def matches(self, fn):
        # TODO: Handle tuple types.
        items = fn.inputs if self.is_input else fn.output
        if self.position is not None:
            if len(items) <= self.position:
                return False
            return self.type == items[self.position].ty
        else:
            return any(self.type == item.ty for item in items)
