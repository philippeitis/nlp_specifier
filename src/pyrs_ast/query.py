from collections import Collection
from math import inf
from typing import List

from . import lib
from .ast_types import TupleType


class QueryField:
    EVAL_COST = inf

    def matches(self, fn: "Fn") -> bool:
        pass


class FnArg(QueryField):
    EVAL_COST = 1e-5

    def __init__(self, xtype, position: int = None, is_input: bool = True):
        self.type = xtype
        self.position = position
        self.is_input = is_input

    def matches(self, fn):
        if not isinstance(fn, lib.Fn):
            return False

        def eq(ty, fn_ty):
            if hasattr(fn_ty, "ty"):
                fn_ty = fn_ty.ty
            else:
                # NeverTy, EmptyTy
                return ty == str(fn_ty)
            if ty == fn_ty:
                return True
            if isinstance(ty, str) and fn_ty is not None:
                if hasattr(fn_ty, "path"):
                    return ty == str(fn_ty.path)
                if hasattr(fn_ty, "name"):
                    return ty == str(fn_ty.name())
            return False

        # TODO: Handle tuple types.
        items = fn.inputs if self.is_input else fn.output

        if isinstance(items, TupleType):
            items = items.elems
        if not isinstance(items, list):
            items = [items]

        if self.position is not None:
            if len(items) <= self.position:
                return False
            return eq(self.type, items[self.position].ty)
        else:
            return any(eq(self.type, item) for item in items)


class Query:
    def __init__(self, fields: List[QueryField], find_types=None):
        self.find_types = find_types
        self.fields = sorted(fields, key=lambda x: x.EVAL_COST)

    def sort_fields(self):
        self.fields = sorted(self.fields, key=lambda x: x.EVAL_COST)

    def matches_item(self, item) -> bool:
        if self.find_types:
            if not isinstance(item, self.find_types):
                return False
        return all(field.matches(item) for field in self.fields)

    def append_field(self, field: QueryField):
        for i, fieldx in enumerate(self.fields):
            if fieldx.EVAL_COST > field.EVAL_COST:
                self.fields.insert(i, field)
                return
        self.fields.append(field)


class Queryable:
    def find_item(self, item_ty, item_path):
        """Return the item at the path with the particular type."""
        raise NotImplementedError()

    def find_matches(self, query: Query, recurse=None) -> Collection:
        """Return all items which match the particular set of queries, in no particular order."""
        raise NotImplementedError()
