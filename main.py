from pyrs_ast.utils import read_ast_from_path, print_ast_docs, print_ast


def main(path):
    ast = read_ast_from_path(path)
    print("PRINTING DOCS")
    print_ast_docs(ast)
    print("PRINTING FILE")
    print_ast(ast)


if __name__ == '__main__':
    main("data/test.rs")

    # Add examples
    # Motivate problems with what is being accomplished
    # problem and solution and reflection - therefore we do this
    # design writeup
    # bert library
    # begin translating trees
    # search around
