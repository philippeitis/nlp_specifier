from pyrs_ast.lib import HasItems, HasAttrs, AstFile


def print_ast_docs(ast: HasItems):
    for item in ast.items:
        if isinstance(item, HasAttrs):
            for section in item.docs.sections():
                print(section.header)
                print(section.body)
        if isinstance(item, HasItems):
            print_ast_docs(item)


def main(path):
    ast = AstFile.from_path(path)
    print("PRINTING DOCS")
    print_ast_docs(ast)
    print("PRINTING FILE")
    print(ast)


if __name__ == '__main__':
    main("data/test.rs")

    # Motivate problems with what is being accomplished
    # problem and solution and reflection - therefore we do this
    # design writeup
    # bert library
    # search around
