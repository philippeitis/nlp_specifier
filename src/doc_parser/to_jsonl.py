import json

from pyrs_ast.lib import HasItems, Fn, AstFile


def sentence_to_jsonl(sentence: str):
    return {"text": sentence, "label": []}


def to_jsonl(ast: HasItems):
    res = []
    for item in ast.items:
        if isinstance(item, HasItems):
            res += to_jsonl(item)
        elif isinstance(item, Fn):
            res += [
                sentence_to_jsonl(sent) for sent in item.docs.sections()[0].sentences
            ]
    return res


def to_jsonl_file(ast: HasItems, path: str):
    with open(path, "w") as f:
        for item in to_jsonl(ast):
            f.write(json.dumps(item))
            f.write("\n")


if __name__ == '__main__':
    to_jsonl_file(AstFile.from_path("../data/test3.rs"), "../data/test3.txt")