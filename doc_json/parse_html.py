import itertools
import logging
from pathlib import Path
from typing import List, Dict, Union, Tuple
from multiprocessing import Pool
import subprocess

from lxml import etree
from lxml.html import parse

from pyrs_ast.docs import Docs
from py_cargo_utils import rustup_home

from pyrs_ast.lib import Crate, Fn, Struct, Method, Mod, AstFile, LexError
from pyrs_ast.scope import Scope

LOGGER = logging.getLogger(__name__)

HEADER = {
    f"h{n}": "#" * n for n in range(1, 7)
}


def peek(it):
    first = next(it)
    return first, itertools.chain([first], it)


def remove_all_spans(item: etree.ElementBase):
    for val in item.findall("span"):
        if not {"since", "notable-traits-tooltip", "notable-traits", "out-of-band"}.isdisjoint(val.classes):
            val.drop_tree()


def remove_mustuse_div(item: etree.ElementBase):
    for val in item.findall("div"):
        if not {"code-attribute"}.isdisjoint(val.classes):
            val.drop_tree()


def remove_all_src_links(item: etree.ElementBase):
    for val in item.findall("a"):
        if "srclink" in val.classes:
            val.drop_tree()
        if val.attrib["href"] == "javascript:void(0)":
            val.drop_tree()


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


def parse_doc(html_doc: etree.ElementBase) -> Docs:
    docs = Docs()
    if html_doc is None:
        return docs

    for item in html_doc:
        h = HEADER.get(item.tag)
        if h and "section-header" in item.classes:
            docs.push_lines(f"{h} {stringify(item)}")
        elif item.tag == "p":
            docs.push_lines(stringify(item))
        elif item.tag == "div" and "example-wrap" in item.classes:
            docs.push_lines(parse_example(item))
        elif item.tag == "ul":
            for sub_item in RList(item).items:
                docs.push_lines("* " + sub_item.strip())
        elif item.tag == "ol":
            for n, sub_item in enumerate(RList(item).items):
                docs.push_lines("{n}. " + sub_item.strip())
        elif item.tag == "blockquote":
            docs.push_lines(f"> {stringify(item)}")
        elif item.tag == "table":
            # Tables not handled. std::mem::size_of demonstrates usage.
            pass
        elif item.tag == "div" and "information" in item.classes:
            # Tooltips are not handled.
            pass
        else:
            LOGGER.debug("Unknown item", item)
    docs.consolidate()
    return docs


def parse_example(doc: etree.ElementBase) -> str:
    return "```CODE```"


class DocStruct(Struct):
    @classmethod
    def from_block(cls, body: etree.ElementBase, scope: Scope = None):
        type_decl = find_with_class(body, "div", "docblock")
        type_decl.drop_tree()

        parent = find_with_class(body, "details", "top-doc")
        if parent is not None:
            doc = find_with_class(parent, "div", "docblock")
            docs = parse_doc(doc)
            parent.drop_tree()
        else:
            docs = Docs()

        struct = cls.from_str(f"{docs}\n{stringify(type_decl)}")
        struct.eat_impls(body)
        return struct

    def eat_impls_old(self, doc: etree.ElementBase, doc_iter) -> List["DocMethod"]:
        items = []
        while True:
            try:
                item, doc_iter = peek(doc_iter)
                if item.tag != "div":
                    return items + self.eat_impl(doc, doc_iter)
                remove_all_spans(item)
                remove_all_src_links(item)
                remove_mustuse_div(item)
                name = stringify(item)
                next(doc_iter)
                item.drop_tree()
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
            items.append(DocMethod.from_str(f"{Docs()}\n{name} {{}}"))

        return items

    def eat_impl(self, doc: etree.ElementBase, doc_iter) -> List["DocMethod"]:
        items = []
        while True:
            try:
                item, doc_iter = peek(doc_iter)
                if item.tag != "details":
                    return items + self.eat_impls_old(doc, doc_iter)
                try:
                    items.append(DocMethod.from_block(item))
                except ValueError:
                    pass
                next(doc_iter)
                item.drop_tree()

            except StopIteration:
                break
        return items

    def eat_impl_dispatch(self, doc: etree.ElementBase) -> List["DocMethod"]:
        try:
            first = next(iter(doc))
            if "method" in first.classes:
                return self.eat_impls_old(doc, iter(doc))
            return self.eat_impl(doc, iter(doc))

        except StopIteration:
            return []

    def eat_impls(self, body):
        impl_block = find_with_class(body, "details", "implementors-toggle")
        if impl_block is not None:
            for doc in find_with_class_iter(impl_block, "div", "impl-items"):
                self.methods += self.eat_impl_dispatch(doc)


class DocPrimitive(DocStruct):
    @classmethod
    def from_block(cls, body: etree.ElementBase, scope: Scope = None):
        type_decl = find_with_class(body, "h1", "fqn")
        remove_all_spans(type_decl)
        remove_all_src_links(type_decl)
        remove_mustuse_div(type_decl)

        parent = find_with_class(body, "details", "top-doc")
        if parent is not None:
            doc = find_with_class(parent, "div", "docblock")
            docs = parse_doc(doc)
            parent.drop_tree()
        else:
            docs = Docs()

        tdcl = f"struct {stringify(type_decl).rsplit(' ', 1)[1]} {{}}"
        type_decl.drop_tree()
        struct = cls.from_str(f"{docs}\n{tdcl}")
        struct.eat_impls(body)
        return struct


class DocFn(Fn):
    @classmethod
    def from_block(cls, body: etree.ElementBase):
        fn_decl_block = find_with_class(body, "pre", "fn")
        remove_all_src_links(fn_decl_block)
        remove_all_spans(fn_decl_block)
        fn_decl = stringify(fn_decl_block) + " {}"
        fn_decl_block.drop_tree()
        parent = find_with_class(body, "details", "rustdoc-toggle")
        if parent is None:
            docs = Docs()
        else:
            doc = find_with_class(parent, "div", "docblock")
            docs = parse_doc(doc)

        return cls.from_str(f"{docs}\n{fn_decl}")


class DocMethod(Method):
    @classmethod
    def from_block(cls, block) -> "DocMethod":
        name = block.find("summary/div/h4")
        remove_all_src_links(name)
        remove_all_spans(name)
        remove_mustuse_div(name)
        name = stringify(name)

        docs = parse_doc(find_with_class(block, "div", "docblock"))
        return cls.from_str(f"{docs}\n{name} {{}}")


DISPATCH = {
    "fn": DocFn.from_block,
    "struct": DocStruct.from_block,
    "primitive": DocPrimitive.from_block,
}


class DocMod(Mod):
    ITEM_PREFIXES = ("struct.", "fn.", "primitive.")

    # , "enum.", "constant.", "macro.", "trait.", "keyword.")

    @classmethod
    def from_doc_path(cls, path: Path, file_hints=None):
        imports = []
        items = []
        ident = path.name

        for child_path in path.iterdir():
            if child_path.is_dir():
                if child_path.name in DocCrate.IGNORE:
                    continue
                item = DocMod.from_doc_path(child_path, file_hints)
                if len(item.items) != 0:
                    items.append(item)
            else:
                if child_path in file_hints:
                    item, imports, lines = file_hints[child_path]

                    if item is None:
                        continue
                    else:
                        imports += AstFile.from_str("\n".join(imports)).items
                        items.append(item)
                else:
                    name = child_path.name
                    if any(name.startswith(prefix) for prefix in cls.ITEM_PREFIXES):
                        file, _, _ = parse_file(child_path)
                        if file:
                            items.append(file)
        return cls(ident, imports + items, [])


class DocCrate(Crate):
    IGNORE = {
        "rust-by-example", "reference", "embedded-book", "edition-guide", "arch", "core_arch",
        "book", "nomicon", "unstable-book", "cargo", "rustc", "implementors", "rustdoc"
    }

    @classmethod
    def from_root_dir(cls, path: Path, pool: Pool=None):
        if path.is_dir() and path.name in DocCrate.IGNORE:
            raise ValueError("Bad path")
        items = []

        pool = pool or Pool(12)

        files = {
            fpath: (item, rs_path, lines) for item, rs_path, lines, fpath in get_all_files_with_pool(path, pool)
        }

        root_scope = Scope()

        target_dir = (path / Path("share/doc/rust/html/")).expanduser()
        for child_path in target_dir.iterdir():
            if child_path.is_dir() and child_path.name not in cls.IGNORE:
                items.append(DocMod.from_doc_path(child_path, files))

        # This doesn't quite work. Again, compiler extension
        # for mod in items:
        #     mod.resolve_imports(root_scope)
        #
        # for mod in items:
        #     mod.register_types(root_scope)

        return cls(items, root_scope)

    @staticmethod
    def get_all_doc_files(path: Path) -> List[Path]:
        if path.is_dir() and path.name in DocCrate.IGNORE:
            return []

        choices = ("struct.", "fn.", "enum.", "primitive.", "constant.", "macro.", "trait.", "keyword.")

        items = []
        for child_path in path.iterdir():
            if child_path.is_dir():
                items += DocCrate.get_all_doc_files(child_path)
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
        "primitive": [],
        "keyword": [],
    }
    for path in paths:
        path_dict[path.name.split(".")[0]].append(path)
    return path_dict


def parse_rs_html(path: Path) -> List[str]:
    # This is intended to be used for reading imports from the source file. However,
    # this may not be necessary if a compiler extension is implemented.
    pass
    # with open(path, "r") as file:
    #     soup = parse(file)
    #     items = soup.find("body/section/div")
    #
    #     rust_body = find_with_class(items, "pre", "rust")
    #     uses = []
    #     for use in find_with_class_iter(rust_body, "span", "kw"):
    #         use_text = text(use)
    #         if "use" in use_text.split(" "):
    #             uses.append(use_text)
    #     return uses


def parse_file(path: Path) -> Tuple[Union[Struct, Fn], List[str], str]:
    with open(path, "r") as file:
        soup = parse(file)
        title = soup.find("head/title")
        if title.text == "Redirection":
            return None, None, None

        if {"rustdoc", "source"}.issubset(soup.find("body").classes):
            return None, None, None
        body = soup.find("body/section")
        # print(path)
        # header = find_with_class(body, "h1", "fqn")
        # srcspan = find_with_class(header, "span", "out-of-band")
        # srclink = find_with_class(srcspan, "a", "srclink")

        try:
            # if srclink is not None:
            #     src, lines = srclink.attrib["href"].split("#")
            #     srcpath = (path.parent / Path(src)).resolve()
            #
            #     return DISPATCH[path.stem.split(".", 1)[0]](body), parse_rs_html(srcpath), lines
            return DISPATCH[path.stem.split(".", 1)[0]](body), None, None
        except LexError:
            return None, None, None


def _get_all_files_st(toolchain_root: Path):
    target_dir = (toolchain_root / Path("share/doc/rust/html/")).expanduser()
    files = files_into_dict(DocCrate.get_all_doc_files(target_dir))
    targets = files["fn"] + files["struct"]
    return [(file, path, rs_path, lines) for (file, rs_path, lines), path in zip(map(parse_file, targets), targets) if
            file]


def get_all_files(toolchain_root: Path, num_processes: int = 12):
    # For debugging purposes
    if num_processes == 1:
        return _get_all_files_st(toolchain_root)

    get_all_files_with_pool(toolchain_root, Pool(num_processes))


def get_all_files_with_pool(toolchain_root: Path, pool: Pool):
    target_dir = (toolchain_root / Path("share/doc/rust/html/")).expanduser()
    files = files_into_dict(DocCrate.get_all_doc_files(target_dir))

    with pool as p:
        targets = files["fn"] + files["struct"] + files["primitive"]
        results = p.map(parse_file, targets)
        return [(file, path, rs_path, lines) for (file, rs_path, lines), path in
                zip(results, targets) if file]


def main(toolchain_root: Path):
    crate = DocCrate.from_root_dir(toolchain_root)
    for file in crate.files:
        print(file)


def choose_random_items(toolchain_root: Path):
    import random
    import webbrowser

    target_dir = (Path(toolchain_root) / Path("share/doc/rust/html/")).expanduser()
    files = files_into_dict(DocCrate.get_all_doc_files(target_dir))

    selected = random.choices(files["struct"], k=25)
    for item in selected:
        webbrowser.open(str(item))
        file, _, _ = parse_file(item)
        if file:
            print(item)
            print(file)
            if isinstance(file, Struct):
                for method in file.methods:
                    print(method)
            input()


def get_toolchains() -> List[Path]:
    return [
        Path(toolchain) for toolchain in get_toolchainsx()
    ]


def profiling(statement: str):
    import cProfile
    import pstats
    cProfile.run(statement, "stats")
    pstats.Stats("stats").sort_stats(pstats.SortKey.TIME).print_stats(20)


if __name__ == "__main__":
    print(get_toolchains())

    profiling("get_all_files(get_toolchains()[0])")
# main(get_toolchains()[0])
# mention that documentation is incomplete spec
# explain that target is verifier, and not all things are supported (eg. sideeffectful operations)
