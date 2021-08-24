from pathlib import Path
from typing import Optional
import logging
import json

import astx
from pyrs_ast.ast_types import Path
from pyrs_ast.lib import HasAttrs, Fn, Fields, Method
from pyrs_ast.scope import Scope

from treevis import PersistentCounter, Node, Edge, call_dot

LOGGER = logging.getLogger(__name__)


def graphviz_escape(s: str) -> str:
    return s.replace("\"", "\\\"").replace("{", "\\{").replace("}", "\\}").replace("<", "\\<").replace(">", "\\>")


def ast_color(name: str) -> str:
    from palette import VERB, NOUN
    if not isinstance(name, str):
        return "black"

    if name.startswith("fn"):
        return VERB
    if name.startswith("struct"):
        return NOUN
    return "black"


def json_to_graph(
        tree,
        parent_id,
        counter: PersistentCounter,
        is_record: bool = False,
        color: str = "black"
):
    if isinstance(tree, str):
        idx = next(counter)
        shape = "record" if is_record else "box"
        tree = tree if is_record else graphviz_escape(tree)
        return [
            Edge(from_id=parent_id, to_id=idx, color=color),
            Node(idx, tree, shape=shape, color=color)
        ]

    items = []
    if isinstance(tree, dict):
        for name, child in tree.items():
            if name == "output":
                child = f"{{ <f0> output|<f1> {graphviz_escape(child)} }}"
                items += json_to_graph(child, parent_id, counter, is_record=True)
                continue
            if name in {"inputs", "fields"}:
                child = f"{{ <f0> {name}|" + "|".join(
                    f"<f{num + 1}> {graphviz_escape(attr)}\l" for num, attr in enumerate(child)) + "}"
                items += json_to_graph(child, parent_id, counter, is_record=True)
                continue
            if name == "attrs":
                child = "{" + "|".join(f"<f{num}> {graphviz_escape(attr)}\l" for num, attr in enumerate(child)) + "}"
            items += json_to_graph(name, parent_id, counter, color=ast_color(name))
            items += json_to_graph(child, counter.peek(), counter, is_record=True)

    if isinstance(tree, list):
        for child in tree:
            items += json_to_graph(child, parent_id, counter, color=ast_color(child))
    return items


def simplify_json(xjson):
    scope = Scope()
    if not isinstance(xjson, dict):
        return

    if "fn" in xjson and "ident" in xjson["fn"]:
        fjson = xjson["fn"]
        fn = Fn(**fjson, scope=scope)
        fjson["inputs"] = [str(inputx) for inputx in fn.inputs]
        fjson["attrs"] = [str(attr) for attr in fn.attrs]
        fjson["output"] = str(fn.output)
        fjson.pop("stmts")
        xjson.pop("fn")
        xjson[f"fn {fjson.pop('ident')}"] = fjson

    if "method" in xjson and "ident" in xjson["method"]:
        fjson = xjson["method"]
        fn = Method(**fjson, scope=scope)
        fjson["inputs"] = [str(inputx) for inputx in fn.inputs]
        fjson["attrs"] = [str(attr) for attr in fn.attrs]
        fjson["output"] = str(fn.output)
        fjson.pop("stmts")
        xjson.pop("method")
        xjson[f"fn {fjson.pop('ident')}"] = fjson

    if "struct" in xjson and "ident" in xjson["struct"]:
        fjson = xjson["struct"]
        xjson.pop("struct")
        xjson[f"struct {fjson.pop('ident')}"] = fjson

    if "impl" in xjson and "self_ty" in xjson["impl"]:
        fjson = xjson["impl"]
        xjson.pop("impl")
        xjson[f"impl {Path(**fjson.pop('self_ty')['path'])}"] = fjson["items"]

    if "path" in xjson:
        path_simple = Path(**xjson["path"])
        xjson["path"] = str(path_simple)

    if "attrs" in xjson and "inputs" not in xjson:
        attrs = HasAttrs(**xjson)
        xjson["attrs"] = [str(attr) for attr in attrs.attrs]

    if "fields" in xjson:
        f = Fields(scope, xjson["fields"])
        xjson["fields"] = [str(field) for field in f]

    for name in list(xjson.keys()):
        if not xjson[name]:
            xjson.pop(name)

    for name, vals in xjson.items():
        if isinstance(vals, dict):
            simplify_json(vals)
        elif isinstance(vals, list):
            for val in vals:
                simplify_json(val)


def line_no(s: str) -> str:
    lines = s.splitlines()
    line_len = len(lines)
    no_width = len(str(line_len - 1))
    return "\n".join(f"[{str(no).zfill(no_width)}] {line}" for no, line in enumerate(lines))


def graph_from_rs_code(code: str, filename: str, root_name="root"):
    xjson = json.loads(astx.ast_from_str(code))
    simplify_json(xjson)

    counter = PersistentCounter()
    root_id = counter.peek()
    items = [Node(root_id, root_name, shape="box")]
    items += json_to_graph(xjson["items"], root_id, counter)

    dot_str = "digraph {\n    " + "\n    ".join(str(item) for item in items) + "\n}\n"
    call_dot(dot_str, filename)


if __name__ == '__main__':
    with open("../data/test5.rs", "r") as f:
        graph_from_rs_code(f.read(), "ast_test5.pdf", "test5.rs")

    # make section 1 intro / problem statement / high level approach
    # diagram of process eg. tokenizer -> parser -> specifier -> search
    # section 1.1. motivate sequence of problems
    # sections 1.2. high level details - methods used and why
    # section 3. specific details
    # mention that documentation is incomplete spec
    # explain that target is verifier, and not all things are supported (eg. sideeffectful operations)
    # squares connecting each component
