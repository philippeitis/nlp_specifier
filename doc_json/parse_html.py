import os
import webbrowser
from pathlib import Path
from typing import List, Dict

import bs4
from bs4 import BeautifulSoup

SINCE = ("span", {"class": "since"})
SRCLINK = ("a", {"class": "srclink"})


def eat_h1(body) -> str:
    header = body.find("h1", {"class": "fqn"})
    to_remove = [SINCE, SRCLINK, ("a", {"href": "javascript:void(0)"})]
    for item, data in to_remove:
        for div in header.find_all(item, data):
            div.decompose()
    text = str(header.text)
    header.decompose()
    return text


def parse_example(doc) -> str:
    for item in doc:
        if isinstance(item, bs4.NavigableString):
            continue
    return "```CODE```"


def to_rust_doc(doc) -> str:
    doc_items = []
    for item in doc:
        if isinstance(item, bs4.NavigableString):
            continue
        if item.name == "p":
            doc_items.append(item.text)
        elif item.name == "div" and "example-wrap" in item.attrs["class"]:
            doc_items.append(parse_example(item))
        elif item.name == "h1" and "section-header" in item.attrs["class"]:
            doc_items.append(f"# {item.text}")
        else:
            try:
                if "stab" in item.attrs["class"]:
                    item.decompose()
                else:
                    print(item, item.attrs["class"])
            except KeyError:
                print("STABBY")
                print(item)
    return "\n".join(doc_items)


def eat_struct_doc(body) -> str:
    items = body.find_all("div", {"class": "docblock"}, recursive=False)
    type_decl = items[0]
    doc = items[1]
    doc_text = to_rust_doc(doc)
    doc.decompose()
    type_decl_text = type_decl.text
    type_decl.decompose()
    return type_decl_text, doc_text


def eat_fn_doc(body) -> str:
    doc = body.find("div", {"class": "docblock"}, recursive=False)
    if not doc:
        parent = body.find("details", {"class": "rustdoc-toggle"})
        if not parent:
            return ""
        doc = parent.find("div", {"class": "docblock"}, recursive=False)

    doc_text = to_rust_doc(doc)
    doc.decompose()
    return doc_text


def eat_impl_item(body) -> list:
    items = []
    for item in body:
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
            items.append(("DOC", to_rust_doc(item)))
            item.decompose()
    return items


def eat_impls(body) -> list:
    items = []
    for doc in body.find_all("div", {"class": "impl-items"}, recursive=False):
        items.append(eat_impl_item(doc))
    return items


def get_all_doc_files(path: Path) -> list:
    if path.is_dir() and path.name in ["rust-by-example",  "reference", "embedded-book", "edition-guide"]:
        return []

    choices = ["struct.", "fn.", "enum.", "constant.", "macro.", "trait."]

    items = []
    for child_path in path.iterdir():
        if child_path.is_dir():
            items += get_all_doc_files(child_path)
        else:
            name = child_path.name
            if any(name.startswith(choice) for choice in choices) and not "arch" in str(path):
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


def parse_file(path: Path):
    with open(path, "r") as file:
        file_name = os.path.basename(path)
        print(file_name)
        soup = BeautifulSoup(file.read(), 'html.parser')
        if soup.find("title").text == "Redirection":
            return
        body = soup.find("section")
        print(eat_h1(body))
        if file_name.split(".", 1)[0] == "struct":
            print(eat_struct_doc(body))
            print("\n".join(str(x) for x in eat_impls(body)))
        elif file_name.split(".", 1)[0] == "fn":
            print(eat_fn_doc(body))


def main():
    TOOLCHAIN_ROOT = "~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/"
    print(os.environ.get("RUSTUP_HOME"))
    # path = os.path.expanduser(HTML_ROOT + "/rust/html/core/num/struct.NonZeroI8.html")
    target_dir = (Path(TOOLCHAIN_ROOT) / Path("share/doc/rust/html/")).expanduser()

    # path = os.path.expanduser(TOOLCHAIN_ROOT + "share/doc/rust/html/core/char/fn.from_u32.html")
    # webbrowser.open(path)

    files = files_into_dict(get_all_doc_files(target_dir))
    counts = {
        path_type: len(items) for path_type, items in files.items()
    }
    for file in files["fn"]:
        parse_file(file)

    exit()


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
