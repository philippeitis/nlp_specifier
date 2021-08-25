import re
from pathlib import Path
from typing import List, Dict, Optional, Union

import bs4
from bs4 import BeautifulSoup

SINCE = ("span", {"class": "since"})
SRCLINK = ("a", {"class": "srclink"})


class RList:
    def __init__(self, doc):
        self.items = []
        for item in doc:
            if isinstance(item, bs4.NavigableString):
                continue
            assert item.name == "li"
            self.items.append(item.text)


class HasDoc:
    HEADER_RE = re.compile('^h([1-6])$')

    def parse_doc(self, doc):
        doc_items = []
        if not doc:
            return ""

        for item in doc:
            if isinstance(item, bs4.NavigableString):
                continue
            match = self.HEADER_RE.match(item.name)
            if match and "section-header" in item.attrs["class"]:
                h = "#" * int(match.group(1))
                doc_items.append(f"{h} {item.text}")
            elif item.name == "p":
                doc_items.append(item.text.replace("\n", " "))
            elif item.name == "div" and "example-wrap" in item.attrs["class"]:
                doc_items.append(self.parse_example(item))
            elif item.name == "ul":
                doc_items += ["* " + sub_item.strip() for sub_item in RList(item).items]
            elif item.name == "ol":
                doc_items += ["{n}. " + sub_item.strip() for n, sub_item in enumerate(RList(item).items)]
            elif item.name == "blockquote":
                doc_items += f"> {item.text}"
            elif item.name == "table":
                # Tables not handled. std::mem::size_of demonstrates usage.
                pass
            elif item.name == "div" and "information" in item.attrs["class"]:
                # Tooltips are not handled.
                pass
            else:
                print("Unknown item", item)

        doc.decompose()
        return "/// " + "\n/// ".join(doc_items)

    def parse_example(self, doc) -> str:
        for item in doc:
            if isinstance(item, bs4.NavigableString):
                continue
        return "```CODE```"


class HasHeader:
    def __init__(self, body):
        header = body.find("h1", {"class": "fqn"})
        to_remove = [SINCE, SRCLINK, ("a", {"href": "javascript:void(0)"})]
        for item, data in to_remove:
            for div in header.find_all(item, data):
                div.decompose()
        text = str(header.text)
        header.decompose()
        self.header = text


class Struct(HasHeader, HasDoc):
    def __init__(self, body):
        super().__init__(body)
        type_decl = body.find("div", {"class": "docblock"})
        parent = body.find("details", {"class": "rustdoc-toggle"})
        items = parent.find("div", {"class": "docblock"})
        doc = items
        self.doc_text = self.parse_doc(doc)
        self.type_decl_text = type_decl.text
        type_decl.decompose()

        self.impls = []
        for doc in body.find_all("div", {"class": "impl-items"}, recursive=False):
            self.eat_impl(doc)

    def eat_impl(self, doc):
        items = []
        for item in doc:
            if item.name == "h4":
                for target_type in [
                    SINCE, SRCLINK,
                    ("a", {"href": "javascript:void(0)"}),
                ]:
                    for target in item.find_all(*target_type):
                        target.decompose()
                items.append(("METHOD", item.text))
                item.decompose()
            if item.name == "div":
                items.append(("DOC", self.parse_doc(item)))
        return items

    def decl(self):
        return self.doc_text + "\n" + self.type_decl_text


class Fn(HasHeader, HasDoc):
    def __init__(self, body):
        super().__init__(body)
        fn_decl_block = body.find("pre", {"class": "fn"})
        for span in fn_decl_block.find_all("span"):
            span.decompose()
        self.fn_decl = fn_decl_block.text
        parent = body.find("details", {"class": "rustdoc-toggle"})
        if not parent:
            self.doc_text = ""
            return
        doc = parent.find("div", {"class": "docblock"}, recursive=False)
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

    choices = ["struct.", "fn.", "enum.", "constant.", "macro.", "trait."]

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
    }
    for path in paths:
        path_dict[str(path.name).split(".")[0]].append(path)
    return path_dict


def parse_file(path: Path) -> Optional[Union[Struct, Fn]]:
    with open(path, "r") as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
        if soup.find("title").text == "Redirection":
            return
        body = soup.find("section")
        return DISPATCH.get(
            #
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
        print(path)
        if file:
            sucesses.append((path, file))

    for path, file in sucesses:
        file_name = path.name
        # print(file_name)
        # print(file.decl())

    print(len(sucesses), counts)


if __name__ == '__main__':
    main()
    # make section 1 intro / problem statement / high level approach
    # diagram of process eg. tokenizer -> parser -> specifier -> search
    # section 1.1. motivate sequence of problems
    # sections 1.2. high level details - methods used and why
    # section 3. specific details
    # mention that documentation is incomplete spec
    # explain that target is verifier, and not all things are supported (eg. sideeffectful operations)
    # squares connecting each component
