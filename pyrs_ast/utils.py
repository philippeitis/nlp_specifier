import jsons as jsons
import json

from pyrs_ast import AstFile
import astx

from pyrs_ast.lib import HasAttrs


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
