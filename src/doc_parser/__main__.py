import time
import logging
from pathlib import Path
import click

DIR_PATH = Path(__file__).parent.resolve()

# This is a bit of a hack for running things in PyCharm
try:
    from doc_parser import Parser, GRAMMAR_PATH, is_quote
except ImportError:
    import sys
    sys.path.insert(0, str(DIR_PATH))
    from doc_parser.doc_parser import Parser, GRAMMAR_PATH, is_quote

TESTCASE_PATH = DIR_PATH / "base_grammar_test_cases.txt"
LOGGER = logging.getLogger(__name__)


# cargo flamegraph
def profiling(statement: str):
    import cProfile
    import pstats
    cProfile.run(statement, "stats")
    pstats.Stats("stats").sort_stats(pstats.SortKey.TIME).print_stats(20)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--path', "-p", default=DIR_PATH / "../data/test3.rs", help='Source file to specify.', type=Path)
def end_to_end(path: Path):
    """Demonstrates entire pipeline from end to end on provided file. Does not write to file."""
    # TODO: Invoke Rust
    pass

@cli.group()
def specify():
    """Creates specifications for a variety of sources."""
    pass


@specify.command("sentence")
@click.argument("sentence")
def specify_sentence(sentence: str):
    """Specifies the sentence, and prints the specification to the console."""
    # TODO: Invoke Rust
    pass

@specify.command("file")
@click.option('--path', "-p", default=DIR_PATH / "../data/test3.rs", help='Source file to specify.', type=Path)
@click.option('--dest', "-d", default=None, type=Path,
              help='Output file path. If not specified, defaults to --path variable, adding _specified suffix.'
              )
def specify_file(path: Path, dest: Path):
    """Specifies the items in the file at path, and writes a copy of the file with specifications included to
    `dest`."""
    # TODO: Invoke Rust
    pass


@specify.command("docs")
@click.option('--path', default=None, help='Path to documentation.', type=Path)
def specify_docs(path: Path):
    """Specifies each item in the documentation at the given path.
    Documentation can be generated using `cargo doc`, or can be downloaded via `rustup`.

    By default, specifies items in toolchain documentation.
    """
    # TODO: Invoke Rust
    pass


@specify.command("testcases")
@click.option('--path', default=TESTCASE_PATH, help='Path to test cases.', type=Path)
def specify_testcases(path: Path):
    """Specifies each newline separated sentence in the provided file."""
    # TODO: Invoke Rust
    pass


def tokenize_all_sents():
    parser = Parser.default()
    sentences = list(
        set([line.strip() for line in Path("./doc_parser/rs_doc_parser/sents.txt").read_text().splitlines()]))
    start = time.time()
    # tokens = [parser.tokenize(sentence) for sentence in sentences]
    sents = parser.stokenize(sentences)
    end = time.time()
    for sent in sents[0:250]:
        print('"' + " ".join(sent.tags) + '",')
    exit()
    # print([(sent.tags, sent.words) for sent in sents[0:20]])
    # print(set(item for sent in sents for item in sent.tags))
    unique_lits = []
    for sent in sents:

        for tag, word in zip(sent.tags, sent.words):
            if tag == "LIT":
                print(" ".join(sent.words))
                print(word)
                break

    print(unique_lits)
    # print("\n".join(str(x.tags) for x in tokens))
    print(end - start)

    start = time.time()
    tokens = parser.stokenize(sentences)
    # tokens = [parser.tokenize(sentence) for sentence in sentences]
    end = time.time()

    # print("\n".join(str(x.tags) for x in tokens))
    print(end - start)


if __name__ == '__main__':
    formatter = logging.Formatter('[%(name)s/%(funcName)s] %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    logging.getLogger().setLevel(logging.WARNING)
    cli()

    # bert library
    # search around

    # TODO: Detect duplicate invocations.
    # TODO: keyword in fn name, capitalization?
    # TODO: similarity metrics (capitalization, synonym distance via wordnet)
    # TODO: Decide spurious keywords

    # TODO: Mechanism to evaluate code quality
    # TODO: Add type to CODE item? eg. CODE_USIZE, CODE_BOOL, CODE_STR, then make CODE accept all of these
    #  std::any::type_name_of_val
    # TODO: allow specifying default value in #[invoke]
    #  eg. #[invoke(str, arg1 = 1usize, arg2 = ?, arg3 = ?)]

    # TODO:
    #  Implement two piece grammar & code-gen (yeet comma / excl / dot rules, fn_calls.py)
    #  Test FNVHasher / alternatives for CFG
    #  Finish porting grammar.py (yeet lemmatizer.py, grammar.py)
    #  1. parse sent into tokens (falliable)
    #  2. parse tokens into trees (infalliable)
    #  3. parse tree in initial type (infalliable)
    #  4. unresolved code blocks (infalliable)
    #  5. resolved code items (falliable)
    #  6. final specification (infalliable)
