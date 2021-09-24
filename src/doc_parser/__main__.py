import time
import logging
from pathlib import Path

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


def tokenize_all_sents():
    parser = Parser.default()
    sentences = list(set([line.strip() for line in Path("./doc_parser/rs_doc_parser/sents.txt").read_text().splitlines()]))
    sents = parser.stokenize(sentences)

    unique_lits = []
    for sent in sents:

        for tag, word in zip(sent.tags, sent.words):
            if tag == "LIT":
                print(" ".join(sent.words))
                print(word)
                break

    print(unique_lits)


if __name__ == '__main__':
    formatter = logging.Formatter('[%(name)s/%(funcName)s] %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    logging.getLogger().setLevel(logging.WARNING)

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
    #  1. parse sent into tokens (falliable)
    #  2. parse tokens into trees (infalliable)
    #  3. parse tree in initial type (infalliable)
    #  4. unresolved code blocks (infalliable)
    #  5. resolved code items (falliable)
    #  6. final specification (infalliable)
