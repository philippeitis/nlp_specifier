from typing import Dict, Union, List

from pyrs_ast.types import Type


# Acts like a type factory, ensuring that only one instance of a type exists for a particular declaration.
class Scope:
    def __init__(self):
        self.named_types = {}
        self.structs = {}
        self.functions: Dict[str, "Fn"] = {}

        self.parent = None
        self.visibility = None

    def add_struct(self, name: str, struct):
        self.structs[name] = struct
        if name in self.named_types:
            self.named_types[name].register_struct(struct)

    def find_function(self, fn: str):
        return self.functions.get(fn)

    def find_type(self, ty: str):
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


class FnArg:
    def __init__(self, xtype, keyword: str = None, position: int = None):
        self.type = xtype
        self.keyword = keyword
        self.position = position
        self.is_input = False

    def matches_fn(self, fn):
        # TODO: Handle tuple types.
        items = fn.inputs if self.is_input else fn.output

        if self.position is not None:
            if len(items) <= self.position:
                return False
            return self.type == items[self.position]
        else:
            return self.type in items


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
