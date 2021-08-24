from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Optional
from subprocess import run
import logging

from nltk import Tree

from palette import tag_color

LOGGER = logging.getLogger(__name__)


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


def call_dot(dot_str: str, filename: str):
    with NamedTemporaryFile() as temp:
        with open(temp.name, 'w') as f:
            f.write(dot_str)

        file_format = Path(filename).suffix.strip(".")

        cmd = f"dot -T{file_format} < {temp.name} > {filename}"
        LOGGER.info(f"Calling graphviz: {cmd}")
        run(cmd, shell=True)


def render_tree(tree: Tree, filename: str):
    dot_str = "digraph { " + "".join(str(item) for item in tree_to_graph(tree)) + "}\n"
    call_dot(dot_str, filename)
