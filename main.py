from pyrs_ast.utils import read_ast_from_path, print_ast_docs, print_ast


def main():
    ast = read_ast_from_path("pyrs_ast/test.rs")
    print("PRINTING DOCS")
    print_ast_docs(ast)
    print("PRINTING FILE")
    print_ast(ast)


if __name__ == '__main__':
    main()

    # Add examples
    # Motivate problems with what is being accomplished
    # problem and solution and reflection - therefore we do this
