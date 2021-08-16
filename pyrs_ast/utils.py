from pyrs_ast import AstFile
from pyrs_ast.lib import HasAttrs, HasItems


def print_ast_docs(ast: HasItems):
    for item in ast.items:
        if isinstance(item, HasAttrs):
            for section in item.docs.sections():
                print(section.header)
                print(section.body)
        if isinstance(item, HasItems):
            print_ast_docs(item)


def print_ast(ast: AstFile):
    for attr in ast.attrs:
        print(attr)

    for item in ast.items:
        print(item)
        print()
