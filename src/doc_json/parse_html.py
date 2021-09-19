import logging
from pathlib import Path
from typing import List, Dict, Union
import json

from pyrs_ast.docs import Docs
from py_cargo_utils import get_toolchains as get_toolchainsx, parse_all_files

from pyrs_ast.lib import Crate, Fn, Struct, Method, Mod, AstFile, LexError
from pyrs_ast.scope import Scope

LOGGER = logging.getLogger(__name__)


class DocStruct(Struct):
    def eat_impls_old(self, doc_iter) -> List["DocMethod"]:
        items = []
        while True:
            try:
                item, doc_iter = peek(doc_iter)
                if item.tag != "div":
                    return items + self.eat_impl(doc_iter)
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
                item = file_hints.get(child_path)

                if item is None:
                    continue
                else:
                    items.append(item)

        return cls(ident, imports + items, [])


class DocCrate(Crate):
    IGNORE = {
        "rust-by-example", "reference", "embedded-book", "edition-guide", "arch", "core_arch",
        "book", "nomicon", "unstable-book", "cargo", "rustc", "implementors", "rustdoc", "src"
    }

    @classmethod
    def from_root_dir(cls, path: Path):
        if path.is_dir() and path.name in DocCrate.IGNORE:
            raise ValueError("Bad path")
        items = []

        target_dir = (path / Path("share/doc/rust/html/")).expanduser()

        files = convert_rs_to_py(parse_all_files(str(target_dir)))

        root_scope = Scope()

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


def get_all_files(toolchain_root: Path) -> Dict[Path, Union[Struct, Fn]]:
    return convert_rs_to_py(parse_all_files(str(toolchain_root)))


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


class Primitive(Struct):
    pass


def load_item(ident, val):
    dispatch = {"struct": Struct, "method": Method, "fn": Fn, "primitive": Primitive}
    try:
        return dispatch[ident](json.loads(val))
    except json.decoder.JSONDecodeError:
        return
    except ValueError:
        return
    except KeyError:
        return
        # print(val)


def convert_rs_to_py(files) -> Dict[Path, Union[Struct, Fn]]:
    new_files = {}
    for file in files:
        name, path, base = file[0:3]
        item = load_item(name, base)

        if item is None:
            continue

        if name == "struct" or name == "primitive":
            item.methods = [load_item("method", item) for item in file[3]]
            item.methods = [method for method in item.methods if method]

        new_files[Path(path)] = item
    return new_files


if __name__ == "__main__":
    import time

    target_dir = str((get_toolchains()[0] / Path("share/doc/rust/html/")).expanduser())
    start = time.time()
    print(len(convert_rs_to_py(parse_all_files(target_dir))))
    end = time.time()
    print(end - start)
    profiling("convert_rs_to_py(parse_all_files(target_dir))")

# mention that documentation is incomplete spec
# explain that target is verifier, and not all things are supported (eg. sideeffectful operations)
