import itertools
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


def peek(it):
    first = next(it)
    return first, itertools.chain([first], it)


def remove_all_spans(item: etree.ElementBase):
    for val in item.findall("span"):
        if "since" in val.classes or "notable-traits-tooltip" in val.classes:
            item.remove(val)


def remove_all_src_links(item: etree.ElementBase):
    for val in item.findall("a"):
        if "srclink" in val.classes:
            item.remove(val)
        if val.attrib["href"] == "javascript:void(0)":
            item.remove(val)


def find_with_class(item: etree.ElementBase, tag: str, class_: str):
    try:
        return next(find_with_class_iter(item, tag, class_))
    except StopIteration:
        return None


def find_with_class_iter(item: etree.ElementBase, tag: str, class_: str):
    for result in item.findall(tag):
        if class_ in result.classes:
            yield result


def text(item: etree.ElementBase) -> str:
    return etree.tostring(item, method="text", encoding="unicode")


def stringify(item: etree.ElementBase) -> str:
    s = ""
    itext = item.text or ""
    if item.tag == "code":
        s += f"`{itext}`"
    else:
        s += itext

    for sub in item.getchildren():
        s += stringify(sub)
        s += sub.tail or ""

    return s


class RList:
    def __init__(self, doc):
        self.items = []
        for item in doc:
            assert item.tag == "li"
            self.items.append(stringify(item))


class HasDoc:
    HEADER = {
        f"h{n}": "#" * n for n in range(1, 7)
    }

    @classmethod
    def parse_doc(cls, html_doc: etree.ElementBase) -> Docs:
        docs = Docs()
        if html_doc is None:
            return docs

        for item in html_doc:
            h = cls.HEADER.get(item.tag)
            if h and "section-header" in item.classes:
                docs.push_line(f"{h} {stringify(item)}")
            elif item.tag == "p":
                text_ = stringify(item)
                if text_:
                    docs.push_line(text_.replace("\n", " "))
                else:
                    docs.push_line("")
            elif item.tag == "div" and "example-wrap" in item.classes:
                docs.push_line(cls.parse_example(item))
            elif item.tag == "ul":
                for sub_item in RList(item).items:
                    docs.push_line("* " + sub_item.strip())
            elif item.tag == "ol":
                for n, sub_item in enumerate(RList(item).items):
                    docs.push_line("{n}. " + sub_item.strip())
            elif item.tag == "blockquote":
                docs.push_line(f"> {stringify(item)}")
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

    @classmethod
    def parse_example(cls, doc: etree.ElementBase) -> str:
        return "```CODE```"


class HasHeader:
    def __init__(self, body: etree.ElementBase):
        header = find_with_class(body, "h1", "fqn")
        remove_all_spans(header)
        remove_all_src_links(header)
        text_ = stringify(header)
        body.remove(header)
        self.header = text_


class Struct(HasHeader, HasDoc):
    def __init__(self, body: etree.ElementBase):
        super().__init__(body)
        type_decl = find_with_class(body, "div", "docblock")
        body.remove(type_decl)
        self.methods = []

        parent = find_with_class(body, "details", "top-doc")
        if parent is not None:
            doc = find_with_class(parent, "div", "docblock")
            self.docs = self.parse_doc(doc)
            body.remove(parent)
        else:
            self.docs = Docs()

        impl_block = find_with_class(body, "details", "implementors-toggle")
        if impl_block is not None:
            for doc in find_with_class_iter(impl_block, "div", "impl-items"):
                self.methods += self.eat_impl_dispatch(doc)

        self.type_decl = stringify(type_decl)

    def eat_impls_old(self, doc: etree.ElementBase, doc_iter) -> List["Method"]:
        items = []
        while True:
            try:
                item, doc_iter = peek(doc_iter)
                if item.tag != "div":
                    return items + self.eat_impl(doc, doc_iter)
                remove_all_spans(item)
                remove_all_src_links(item)
                name = stringify(item)
                next(doc_iter)
                doc.remove(item)
            except StopIteration:
                break

            try:
                item, doc_iter = peek(doc_iter)
                if "item-info" in item.classes:
                    next(doc_iter)
                else:
                    raise ValueError(str(set(item.classes)))
            except StopIteration:
                pass
            items.append(Method(name, Docs()))

        return items

    def eat_impl(self, doc: etree.ElementBase, doc_iter) -> List["Method"]:
        items = []
        while True:
            try:
                item, doc_iter = peek(doc_iter)
                if item.tag != "details":
                    return items + self.eat_impls_old(doc, doc_iter)
                items.append(Method.from_block(item))
                next(doc_iter)
                doc.remove(item)

            except StopIteration:
                break
        return items

    def eat_impl_dispatch(self, doc: etree.ElementBase) -> List["Method"]:
        try:
            first = next(iter(doc))
            if "method" in first.classes:
                return self.eat_impls_old(doc, iter(doc))
            return self.eat_impl(doc, iter(doc))

        except StopIteration:
            return []

    def decl(self) -> str:
        return f"{self.docs}\n{self.type_decl}"


class Fn(HasHeader, HasDoc):
    def __init__(self, body: etree.ElementBase):
        super().__init__(body)
        fn_decl_block = find_with_class(body, "pre", "fn")
        for span in fn_decl_block.findall("span"):
            fn_decl_block.remove(span)
        self.fn_decl = stringify(fn_decl_block)
        body.remove(fn_decl_block)
        parent = find_with_class(body, "details", "rustdoc-toggle")
        if parent is None:
            self.docs = Docs()
            return
        doc = find_with_class(parent, "div", "docblock")
        self.docs = self.parse_doc(doc)

    def decl(self) -> str:
        return f"{self.docs}\n{self.fn_decl} {{}}"


class Method(HasDoc):
    def __init__(self, name: str, docs: Docs):
        self.name = name
        self.docs = docs

    @classmethod
    def from_block(cls, block) -> "Method":
        name = block.find("summary/div")
        remove_all_src_links(name)
        remove_all_spans(name)
        name = stringify(name)
        docs = cls.parse_doc(find_with_class(block, "div", "docblock"))
        return cls(name, docs)

    def decl(self) -> str:
        return f"{self.docs}\n{self.name}"


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


def get_all_files(toolchain_root: Path):
    target_dir = (toolchain_root / Path("share/doc/rust/html/")).expanduser()
    files = files_into_dict(get_all_doc_files(target_dir))
    with Pool(8) as p:
        targets = files["fn"] + files["struct"]
        return [(file, path) for file, path in zip(p.map(parse_file, targets), targets) if file]


def main(toolchain_root: Path):
    for file, path in get_all_files(toolchain_root):
        print(path)
        print(file.decl())
        if isinstance(file, Struct):
            for impl in file.methods:
                print(impl.decl())


def choose_random_items(toolchain_root: Path):
    import random
    import webbrowser

    target_dir = (Path(toolchain_root) / Path("share/doc/rust/html/")).expanduser()
    files = files_into_dict(get_all_doc_files(target_dir))

    selected = random.choices(files["struct"], k=25)
    for item in selected:
        webbrowser.open(str(item))
        file = parse_file(item)
        if file:
            print(item)
            print(file.decl())
            if isinstance(file, Struct):
                for impl in file.methods:
                    print(impl.decl())
            input()


def get_toolchains() -> List[Path]:
    root = Path(rustup_home()) / Path("toolchains")
    paths = subprocess.run(["rustup", "toolchain", "list"], capture_output=True).stdout.splitlines()
    return [
        root / Path(p.removesuffix(b" (default)").strip().decode("utf-8"))
        for p in paths
    ]


if __name__ == '__main__':
    main(get_toolchains()[0])

    # choose_random_items(get_toolchains()[0])
    # make section 1 intro / problem statement / high level approach
    # diagram of process eg. tokenizer -> parser -> specifier -> search
    # section 1.1. motivate sequence of problems
    # sections 1.2. high level details - methods used and why
    # section 3. specific details
    # mention that documentation is incomplete spec
    # explain that target is verifier, and not all things are supported (eg. sideeffectful operations)
    # squares connecting each component
