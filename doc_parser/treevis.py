from enum import Enum
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Optional
from subprocess import run
import logging

from nltk import Tree

from palette import tag_color

LOGGER = logging.getLogger(__name__)


class GraphvizSubcommand(str, Enum):
    DOT = "dot"
    CIRCO = "circo"


class PersistentCounter:
    def __init__(self):
        self.counter = 0

    def peek(self) -> int:
        return self.counter

    def __next__(self) -> int:
        self.counter += 1
        return self.counter

    def __iter__(self):
        return self


class HasProp:
    PROPS = tuple()

    def prop_str(self) -> str:
        props = [(prop, getattr(self, prop)) for prop in self.PROPS]
        return " ".join(f"{key}=\"{val}\"" for key, val in props if val)


class Edge(HasProp):
    __slots__ = ["color", "arrowhead", "to_id", "from_id", "style"]
    PROPS = ("color", "arrowhead", "style")

    def __init__(self, from_id, to_id, color, arrowhead="none", style=None):
        self.color = color
        self.arrowhead = arrowhead
        self.from_id = from_id
        self.to_id = to_id
        self.style = style

    def __str__(self):
        return f"{self.from_id} -> {self.to_id} [{self.prop_str()}];"


class Node(HasProp):
    __slots__ = ["label", "shape", "fontcolor", "color", "idx"]
    PROPS = ("label", "shape", "color", "fontcolor")

    def __init__(self, idx, label, shape, color=None, fontcolor=None):
        self.idx = idx
        self.label = label
        self.shape = shape
        self.color = color
        self.fontcolor = fontcolor or color

    def __str__(self):
        return f"{self.idx} [{self.prop_str()}];"


def tree_to_graph(tree: Tree, counter: Optional[PersistentCounter] = None, leaf_color: Optional[str] = None):
    counter = counter or PersistentCounter()
    idx = counter.peek()

    if isinstance(tree, str):
        return [Node(idx, tree, shape="box", color=leaf_color)]

    pos = tree.label()
    color = tag_color(pos)
    items = [Node(idx, pos, shape="none", fontcolor=color)]

    for child in tree:
        child_id = next(counter)
        # Inherit color
        style = None
        child_color = color
        if isinstance(child, str):
            pass
        else:
            # Edge between different node styles.
            if tag_color(child.label()) != color:
                child_color = "#000000"
            style = "bold"
        items.append(Edge(from_id=idx, to_id=child_id, color=child_color, style=style))

        items += tree_to_graph(child, counter, color)
    return items


def call_dot(dot_str: str, filename: str, module: GraphvizSubcommand = GraphvizSubcommand.DOT):
    with NamedTemporaryFile() as temp:
        with open(temp.name, 'w') as f:
            f.write(dot_str)

        file_format = Path(filename).suffix.strip(".")

        cmd = f"{module.value} -T{file_format} < {temp.name} > {filename}"
        LOGGER.info(f"Calling graphviz: {cmd}")
        run(cmd, shell=True)


def render_tree(tree: Tree, filename: str):
    dot_str = "digraph { " + "\n".join(str(item) for item in tree_to_graph(tree)) + "}\n"
    print(dot_str)
    call_dot(dot_str, filename)


def render_production_rule():
    root = "OBJ"
    rules = "PRP | DT MNN | MNN | CODE | LIT | DT LIT | OBJ OP OBJ | FNCALL | DT FNCALL | DT VBG MNN | PROP_OF OBJ".split(
        "|")
    rules = [rule.strip() for rule in rules]
    tree = Tree(root, rules)
    print(tree)
    render_tree(tree, "./tree.pdf")


if __name__ == '__main__':
    render_production_rule()

    s = """digraph {
    rankdir="LR";
    graph [nodesep="0.1", ranksep="0.5",style="invis"];
    mindist="0.4";
    0 [label="OBJ" shape="circle" color="#00AA00" fontcolor="#00AA00"];
    splines=ortho;
    
    subgraph cluster1 {
        rank="same"
        1 [label="PRP" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
        2 [label="(DT)?   MNN" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
        3 [label="CODE" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
        4 [label="(DT)?   LIT" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
        5 [label="OBJ   OP   OBJ" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
        6 [label="(DT)?   FNCALL" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
        7 [label="DT   VBG   MNN" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
        8 [label="PROP_OF   OBJ" shape="box" color="#00AA00" fontcolor="#00AA00" width=2];
    }
    
    0 -> {1, 2, 3, 4, 4, 5, 6, 7, 8} [color="#00AA00" arrowhead="normal"];

}
"""
    call_dot(s, "tree.pdf", GraphvizSubcommand.DOT)
