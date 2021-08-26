from pathlib import Path
from typing import List, Dict, Optional, Union

from lxml import etree
from lxml.html import parse

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

    def parse_doc(self, doc):
        doc_items = []
        if doc is None:
            return ""

        for item in doc:
            h = self.HEADER.get(item.tag)
            if h and "section-header" in item.classes:
                doc_items.append(f"{h} {text(item)}")
            elif item.tag == "p":
                text_ = text(item)
                if text_:
                    doc_items.append(text_.replace("\n", " "))
                else:
                    doc_items.append("")
            elif item.tag == "div" and "example-wrap" in item.classes:
                doc_items.append(self.parse_example(item))
            elif item.tag == "ul":
                doc_items += ["* " + sub_item.strip() for sub_item in RList(item).items]
            elif item.tag == "ol":
                doc_items += ["{n}. " + sub_item.strip() for n, sub_item in enumerate(RList(item).items)]
            elif item.tag == "blockquote":
                doc_items += f"> {text(item)}"
            elif item.tag == "table":
                # Tables not handled. std::mem::size_of demonstrates usage.
                pass
            elif item.tag == "div" and "information" in item.classes:
                # Tooltips are not handled.
                pass
            else:
                print("Unknown item", item)

        return "/// " + "\n/// ".join(doc_items)

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
            self.doc_text = self.parse_doc(doc)
            self.impls = []
            for doc in find_with_class_iter(body, "div", "impl-items"):
                self.eat_impl(doc)
            body.remove(parent)
        else:
            self.doc_text = ""

        self.type_decl_text = text(type_decl)

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
        return self.doc_text + "\n" + self.type_decl_text


class Fn(HasHeader, HasDoc):
    def __init__(self, body):
        super().__init__(body)
        fn_decl_block = find_with_class(body, "pre", "fn")
        for span in fn_decl_block.findall("span"):
            span.decompose()
        self.fn_decl = text(fn_decl_block)
        parent = find_with_class(body, "details", "rustdoc-toggle")
        if parent is None:
            self.doc_text = ""
            return
        doc = find_with_class(parent, "div", "docblock")
        self.doc_text = self.parse_doc(doc)

    def decl(self):
        return self.doc_text + "\n" + self.fn_decl + " {}"


DISPATCH = {
    "fn": Fn,
    "struct": Struct,
}


def get_all_doc_files(path: Path) -> list:
    if path.is_dir() and path.name in ["rust-by-example", "reference", "embedded-book", "edition-guide"]:
        return []

    choices = ["struct.", "fn.", "enum.", "constant.", "macro.", "trait.", "keyword."]

    items = []
    for child_path in path.iterdir():
        if child_path.is_dir():
            items += get_all_doc_files(child_path)
        else:
            name = child_path.name
            if any(name.startswith(choice) for choice in choices) and "arch" not in str(path):
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
        path_dict[str(path.name).split(".")[0]].append(path)
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


def main():
    TOOLCHAIN_ROOT = "~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/"
    target_dir = (Path(TOOLCHAIN_ROOT) / Path("share/doc/rust/html/")).expanduser()
    # webbrowser.open(path)

    files = files_into_dict(get_all_doc_files(target_dir))
    counts = {
        path_type: len(items) for path_type, items in files.items()
    }

    sucesses = []
    for path in files["fn"] + files["struct"]:
        file = parse_file(path)
        if file:
            sucesses.append((path, file))

    for path, file in sucesses:
        print(path)
        print(file.decl())

    print(len(sucesses), counts)


def profiling(statement: str):
    import cProfile
    import pstats
    cProfile.run(statement, "stats")
    pstats.Stats("stats").sort_stats(pstats.SortKey.TIME).print_stats(20)


if __name__ == '__main__':
    import cProfile

    profiling("main()")
    # make section 1 intro / problem statement / high level approach
    # diagram of process eg. tokenizer -> parser -> specifier -> search
    # section 1.1. motivate sequence of problems
    # sections 1.2. high level details - methods used and why
    # section 3. specific details
    # mention that documentation is incomplete spec
    # explain that target is verifier, and not all things are supported (eg. sideeffectful operations)
    # squares connecting each component
