from pathlib import Path
from typing import List, Dict, Optional, Union
from multiprocessing import Pool
import subprocess

from lxml import etree
from lxml.html import parse

from pyrs_ast.docs import Docs
from py_cargo_utils import rustup_home

SINCE = ("span", {"class": "since"})
SRCLINK = ("a", {"class": "srclink"})


def find_all_spans(item):
    for val in item.findall("span"):
        if "since" in val.classes:
            item.remove(val)


def find_all_src_link(item):
    for val in item.findall("a"):
        if "srclink" in val.classes:
            item.remove(val)
        if val.attrib["href"] == "javascript:void(0)":
            item.remove(val)


def find_with_class(item, tag: str, class_: str):
    try:
        return next(find_with_class_iter(item, tag, class_))
    except StopIteration:
        return None


def find_with_class_iter(item, tag: str, class_: str):
    for result in item.findall(tag):
        if class_ in result.classes:
            yield result


def text(item):
    return etree.tostring(item, method="text", encoding="unicode")


class RList:
    def __init__(self, doc):
        self.items = []
        for item in doc:
            assert item.tag == "li"
            self.items.append(text(item))


class HasDoc:
    HEADER = {
        f"h{n}": "#" * n for n in range(1, 7)
    }

    def parse_doc(self, html_doc) -> Docs:
        docs = Docs()
        if html_doc is None:
            return docs

        for item in html_doc:
            h = self.HEADER.get(item.tag)
            if h and "section-header" in item.classes:
                docs.push_line(f"{h} {text(item)}")
            elif item.tag == "p":
                text_ = text(item)
                if text_:
                    docs.push_line(text_.replace("\n", " "))
                else:
                    docs.push_line("")
            elif item.tag == "div" and "example-wrap" in item.classes:
                docs.push_line(self.parse_example(item))
            elif item.tag == "ul":
                for sub_item in RList(item).items:
                    docs.push_line("* " + sub_item.strip())
            elif item.tag == "ol":
                for n, sub_item in enumerate(RList(item).items):
                    docs.push_line("{n}. " + sub_item.strip())
            elif item.tag == "blockquote":
                docs.push_line(f"> {text(item)}")
            elif item.tag == "table":
                # Tables not handled. std::mem::size_of demonstrates usage.
                pass
            elif item.tag == "div" and "information" in item.classes:
                # Tooltips are not handled.
                pass
            else:
                print("Unknown item", item)
        docs.consolidate()
        return docs

    def parse_example(self, doc) -> str:
        return "```CODE```"


class HasHeader:
    def __init__(self, body):
        header = find_with_class(body, "h1", "fqn")
        find_all_spans(header)
        find_all_src_link(header)
        text_ = text(header)
        body.remove(header)
        self.header = text_


class Struct(HasHeader, HasDoc):
    def __init__(self, body):
        super().__init__(body)
        type_decl = find_with_class(body, "div", "docblock")
        body.remove(type_decl)

        parent = find_with_class(body, "details", "rustdoc-toggle")
        if parent is not None:
            doc = find_with_class(parent, "div", "docblock")
            self.docs = self.parse_doc(doc)
            self.impls = []
            for doc in find_with_class_iter(body, "div", "impl-items"):
                self.eat_impl(doc)
            body.remove(parent)
        else:
            self.docs = Docs()

        self.type_decl = text(type_decl)

    def eat_impl(self, doc):
        items = []
        for item in doc:
            if item.tag == "h4":
                find_all_spans(item)
                find_all_src_link(item)
                items.append(("METHOD", text(item)))
            if item.tag == "div":
                items.append(("DOC", self.parse_doc(item)))
            doc.remove(item)
        return items

    def decl(self):
        return f"{self.docs}\n{self.type_decl}"


class Fn(HasHeader, HasDoc):
    def __init__(self, body):
        super().__init__(body)
        fn_decl_block = find_with_class(body, "pre", "fn")
        for span in fn_decl_block.findall("span"):
            span.decompose()
        self.fn_decl = text(fn_decl_block)
        parent = find_with_class(body, "details", "rustdoc-toggle")
        if parent is None:
            self.docs = Docs()
            return
        doc = find_with_class(parent, "div", "docblock")
        self.docs = self.parse_doc(doc)

    def decl(self):
        return f"{self.docs}\n{self.fn_decl} {{}}"


DISPATCH = {
    "fn": Fn,
    "struct": Struct,
}


def get_all_doc_files(path: Path) -> List[Path]:
    if path.is_dir() and path.name in {
        "rust-by-example", "reference", "embedded-book", "edition-guide", "arch", "core_arch",
        "book", "nomicon", "unstable-book", "cargo", "rustc", "implementors", "rustdoc"
    }:
        return []

    choices = ("struct.", "fn.", "enum.", "constant.", "macro.", "trait.", "keyword.")

    items = []
    for child_path in path.iterdir():
        if child_path.is_dir():
            items += get_all_doc_files(child_path)
        else:
            name = child_path.name
            if any(name.startswith(choice) for choice in choices):
                items.append(child_path)
            # can also have trait or macro
    return items


def files_into_dict(paths: List[Path]) -> Dict[str, List[Path]]:
    path_dict = {
        "fn": [],
        "struct": [],
        "enum": [],
        "constant": [],
        "macro": [],
        "trait": [],
        "keyword": [],
    }
    for path in paths:
        path_dict[path.name.split(".")[0]].append(path)
    return path_dict


def parse_file(path: Path) -> Optional[Union[Struct, Fn]]:
    with open(path, "r") as file:
        soup = parse(file)
        title = soup.find("head/title")
        if title.text == "Redirection":
            return
        body = soup.find("body/section")
        return DISPATCH.get(
            path.stem.split(".", 1)[0],
            # Default case: return None
            lambda *args, **kwargs: None
        )(body)


def main(toolchain_root: Path):
    target_dir = (toolchain_root / Path("share/doc/rust/html/")).expanduser()

    files = files_into_dict(get_all_doc_files(target_dir))
    counts = {
        path_type: len(items) for path_type, items in files.items()
    }

    with Pool(8) as p:
        targets = files["fn"] + files["struct"]
        successes = [(file, path) for file, path in zip(p.map(parse_file, targets), targets) if file]

    for file, path in successes:
        print(path)
        print(file.decl())

    print(len(successes), counts)


def choose_random_items(toolchain_root: Path):
    import random
    import webbrowser

    target_dir = (Path(toolchain_root) / Path("share/doc/rust/html/")).expanduser()
    files = files_into_dict(get_all_doc_files(target_dir))

    selected = random.choices(files["fn"] + files["struct"], k=25)
    for item in selected:
        file = parse_file(item)
        if file:
            print(item)
            print(file.decl())
            webbrowser.open(str(item))
            input()


def get_toolchains() -> List[Path]:
    root = Path(rustup_home()) / Path("toolchains")
    paths = subprocess.run(["rustup", "toolchain", "list"], capture_output=True).stdout.splitlines()
    return [
        root / Path(p.removesuffix(b" (default)").strip().decode("utf-8"))
        for p in paths
    ]


if __name__ == '__main__':
    choose_random_items(get_toolchains()[0])
    # make section 1 intro / problem statement / high level approach
    # diagram of process eg. tokenizer -> parser -> specifier -> search
    # section 1.1. motivate sequence of problems
    # sections 1.2. high level details - methods used and why
    # section 3. specific details
    # mention that documentation is incomplete spec
    # explain that target is verifier, and not all things are supported (eg. sideeffectful operations)
    # squares connecting each component
