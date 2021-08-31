class UseGlob:
    def __init__(self):
        pass

    def get_from_scope(self, scope):
        return [(fn.ident, fn) for fn in scope.functions.values()] + [(s.ident, s) for s in scope.structs.values()]

    def __str__(self):
        return "*"


class UseGroup:
    def __init__(self, tree):
        self.items = []
        for item in tree:
            self.items.append(UseTree(item))

    def get_from_scope(self, scope):
        items = []
        for item in self.items:
            items += item.get_from_scope(scope)
        return items

    def __str__(self):
        return "{" + ", ".join(str(item) for item in self.items) + "}"


class UseRename:
    def __init__(self, tree):
        self.ident = tree["ident"]
        self.rename = tree["rename"]

    def get_from_scope(self, scope):
        return [(self.rename, scope.find_item([self.ident]))]

    def __str__(self):
        return f"{self.ident} as {self.rename}"


class UsePath:
    def __init__(self, tree):
        self.ident = tree["ident"]
        self.tree = UseTree(tree["tree"]).use

    def get_from_scope(self, scope):
        sub = scope.modules[self.ident]
        return self.tree.get_from_scope(sub)

    def __str__(self):
        return f"{self.ident}::{self.tree}"


class UseName:
    def __init__(self, item):
        self.ident = item

    def get_from_scope(self, scope):
        return [(self.ident, scope.find_item([self.ident]))]

    def __str__(self):
        return self.ident


class UseTree:
    def __init__(self, item):
        if item == "*":
            self.use = UseGlob
            return
        key, val = next(iter(item.items()))
        self.use = DISPATCH[key](val)

    def get_from_scope(self, scope):
        return self.use.get_from_scope(scope)

    def __str__(self):
        return str(self.use)


DISPATCH = {
    "rename": UseRename,
    "group": UseGroup,
    "path": UsePath,
    "ident": UseName,
}
