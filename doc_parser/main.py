from collections import defaultdict
import logging
from itertools import chain
from typing import Collection

from nltk import Tree
from spacy.tokens import Doc

from doc_json.parse_html import DocStruct, DocFn
from pyrs_ast.lib import LitAttr, Fn, HasItems, Crate, Struct, Mod, Const
from pyrs_ast.query import Query, FnArg
from pyrs_ast.scope import Scope
from pyrs_ast import AstFile

from doc_parser import Parser, GRAMMAR_PATH, is_quote
from fn_calls import InvocationFactory, Invocation
from grammar import Specification, generate_constructor_from_grammar
from nlp_query import query_from_sentence, Phrase, Word

LOGGER = logging.getLogger(__name__)


def tree_references_fn(fn: Fn, tree: Tree) -> bool:
    """Determines whether the tree references the provided fn."""
    fn_name = f"{fn.ident}"
    if isinstance(tree, str):
        label = tree
    else:
        label: str = tree.label()

    if label.startswith("FN_"):
        name, _ = label.rsplit("_", 1)
        _, name = name.split("_", 1)

        if name == fn_name:
            return True

    if isinstance(tree, str):
        return False

    return any(tree_references_fn(fn, child) for child in tree)


def tree_contains_fn_call(tree: Tree) -> bool:
    """Determines whether the tree references the provided fn."""
    if isinstance(tree, str):
        return tree == "FNCALL"

    if tree.label() == "FNCALL":
        return True

    return any(tree_contains_fn_call(child) for child in tree)


def apply_specifications(fn: Fn, parser: Parser, scope: Scope, invoke_factory):
    """Creates a specification for the function, using all available sentences.
    Each sentence is cross-referenced with the grammar to form a syntax tree. If a syntax tree can be formed,
    this is treated as a valid specification, and is compiled into the Prusti annotation format.

    Prefers non-self-referential trees wherever possible.

    If no tree can be found for a particular sentence, no specification is added.
    """

    if not fn.should_specify():
        LOGGER.info(f"fn {fn.ident} is not annotated with #[specify], and will not be specified.")
        return

    fn_idents = set(ty.ident for ty in fn.inputs)
    sections = fn.docs.sections()
    for section in sections:
        if section.header is not None:
            continue
        LOGGER.info(f"Specifying documentation section {section.header} of {fn.ident}")
        for sentence in section.sentences:
            try:
                skipped_self_ref = []
                skipped_fn_call = []
                parse_it = parser.parse_tree(sentence, idents=fn_idents)
                spec = None
                for tree in parse_it:
                    if tree_references_fn(fn, tree):
                        LOGGER.info("Skipping tree due to self-reference.")
                        skipped_self_ref.append(tree)
                    elif tree_contains_fn_call(tree):
                        LOGGER.info("Skipping tree due to self-reference.")
                        skipped_fn_call.append(tree)
                    else:
                        LOGGER.info("Found tree without self-reference, applying.")
                        spec = Specification(tree, invoke_factory).as_spec()
                        break

                if spec is None:
                    LOGGER.info("Attempting to use specification with function call.")
                    spec = Specification(next(iter(skipped_fn_call)), invoke_factory).as_spec()
                    # spec = Specification(next(iter(skipped)), invoke_factory).as_spec()

                attr = LitAttr(spec)

                LOGGER.info(f"[{sentence}] was transformed into the following specification: {attr}")
                fn.attrs.append(attr)
            except ValueError as v:
                LOGGER.error(f"While specifying [{sentence}], error occurred: {v}")
                LOGGER.info(
                    f"[{sentence}] has the following tags: {parser.tokenize(sentence, idents=fn_idents).tags}"
                )
            except StopIteration as s:
                LOGGER.info(f"No specification could be generated for [{sentence}]")
                LOGGER.info(
                    f"[{sentence}] has the following tags: {parser.tokenize(sentence, idents=fn_idents).tags}"
                )
                query = query_from_sentence(sentence, parser)
                LOGGER.info("Found phrases: " + ", ".join(str(x) for x in query.fields))
                for fn in scope.find_fn_matches(query):
                    LOGGER.info(f"Found match: {fn.sig_str()}")


def specify_item(item: HasItems, parser: Parser, scope: Scope, invoke_factory):
    if isinstance(item, Mod) and item.items is None:
        return

    for sub_item in item.value_iter():
        if isinstance(sub_item, Fn):
            apply_specifications(sub_item, parser, scope, invoke_factory)
        elif isinstance(sub_item, HasItems):
            specify_item(sub_item, parser, scope, invoke_factory)


def find_specifying_sentence(fn: Fn, parser: Parser, invoke_factory: InvocationFactory, word_replacements,
                             sym_replacements):
    """Finds the sentence that specifies the function, and adds it to InvokeFactory.
    NOTE: Currently, the first sentence is treated as the specifying sentence. This is obviously not a robust
    metric, and should be updated.

    Factors which might be considered could be the function's name (for instance, Vec::remove), the number of unique
    idents, incidence of important words.
    """
    sections = fn.docs.sections()
    fn_idents = set(ty.ident for ty in fn.inputs)
    LOGGER.info(f"Searching for explicit invocations for fn {fn.ident}")

    for attr in fn.attrs:
        if str(attr.ident) == "invoke":
            invoke = str(attr.tokens[1].val.strip("\" "))
            LOGGER.info(f"Found invocation [{invoke}] for fn {fn.ident}")
            invoke_factory.add_invocation(fn, Invocation.from_sentence(fn, invoke))

    for section in sections:
        if section.header is not None:
            continue
        LOGGER.info(f"Determining descriptive sentence for {fn.ident}")
        descriptive_sentence = section.sentences[0]
        sent = parser.tokenize(descriptive_sentence, idents=fn_idents)
        for sym, word in zip(sent.tags, sent.words):
            if is_quote(word):
                continue

            word_replacements[word].add(sym)
            sym_replacements[sym].add(word)

        invoke_factory.add_fuzzy_invocation(fn, sent.tags, sent.words)


def populate_grammar_helper(item: HasItems, parser: Parser, invoke_factory, word_replacements, sym_replacements):
    if isinstance(item, Mod) and item.items is None:
        return

    for sub_item in item.value_iter():
        if isinstance(sub_item, Fn):
            find_specifying_sentence(sub_item, parser, invoke_factory, word_replacements, sym_replacements)
        elif isinstance(sub_item, HasItems):
            populate_grammar_helper(sub_item, parser, invoke_factory, word_replacements, sym_replacements)


def generate_grammar(ast, helper_fn=populate_grammar_helper):
    """Iterates through all items in the ast, adds relevant invocations from functions into the grammar,
    and returns a grammar with all invocations, as well as an InvocationFactory to dynamically dispatch
    a particular invocation to the relevant constructor.
    """

    word_replacements = defaultdict(set)
    sym_replacements = defaultdict(set)
    invoke_factory = InvocationFactory(generate_constructor_from_grammar)
    helper_fn(ast, Parser.default(), invoke_factory, word_replacements, sym_replacements)

    grammar = invoke_factory.grammar()
    replacement_grammar = ""

    if "PRP$" in sym_replacements:
        sym_replacements["PRPS"] = sym_replacements.pop("PRP$")

    word_filter = lambda word: not set(",`!.();:*<>=+-?&%'\"\\[]{}#").isdisjoint(word)
    for word, syms in word_replacements.items():
        if word_filter(word):
            continue
        if "PRP$" in syms:
            syms.remove("PRP$")
            syms.add("PRP")

        syms = " | ".join(f"\"{word}_{sym}\"" for sym in syms)
        grammar = grammar.replace(f"\"{word}\"", f"WD_{word}")
        replacement_grammar += f"WD_{word} -> {syms}\n"
    for sym, words in sym_replacements.items():
        if sym in {"COMMA", "DOT", "EXCL", "-LRB-", "-RRB-", ":", ".", ",", "$", "(", ")"}:
            continue
        syms = " | ".join(f"\"{word}_{sym}\"" for word in words if not word_filter(word))
        # grammar = grammar.replace(f"{sym} -> \"{sym}\"", "")
        replacement_grammar += f"{sym} -> {syms} | \"{sym}\"\n"

    with open(GRAMMAR_PATH) as f:
        full_grammar = f.read() + "\n# FUNCTION INVOCATIONS\n\n" + grammar + "\n\n# Word Replacements\n\n" + replacement_grammar

    return full_grammar, invoke_factory


def end_to_end_demo():
    """Demonstrates entire pipeline from end to end."""
    ast = AstFile.from_path("../data/test3.rs")
    grammar, invoke_factory = generate_grammar(ast)

    parser = Parser(grammar)
    print(grammar)
    specify_item(ast, parser, ast.scope, invoke_factory)
    print(ast)


def invoke_helper(invocations: Collection, invocation_triples=None, use_invokes=False):
    """Demonstrates creation of invocations from specifically formatted strings, as well as usage."""

    from nltk import Tree
    from grammar import UnsupportedSpec

    class FnMock:
        def __init__(self, ident: str):
            self.ident = ident

    def populate_grammar_helper(sentences, parser: Parser, invoke_factory, word_replacements, sym_replacements):
        for sentence in sentences:
            if isinstance(sentence, tuple):
                invoke_factory.add_invocation(
                    FnMock(sentence[0]),
                    Invocation.from_sentence(FnMock(sentence[0]), sentence[1])
                )
                sentence = sentence[2]

            sent = parser.tokenize(sentence)
            for sym, word in zip(sent.tags, sent.words):
                if is_quote(word):
                    continue
                word_replacements[word].add(sym)
                sym_replacements[sym].add(word)

    invocation_triples = invocation_triples or []
    if use_invokes:
        grammar, factory = generate_grammar(invocation_triples + invocations, populate_grammar_helper)
        parser = Parser(grammar)
    else:
        parser = Parser.default()
        factory = None

    ntrees = 0
    nspecs = 0
    num_sents = len(invocations) + len(invocation_triples)
    successful_sents = 0
    for sentence in chain(invocations, (sentence for _, _, sentence in invocation_triples)):
        print("=" * 80)
        print("Sentence:", sentence)
        print("    Tags:", parser.tokenize(sentence).tags)
        print("=" * 80)
        specs = []
        trees = []
        try:
            for tree in parser.parse_tree(sentence, attach_tags=use_invokes):
                print(tree)
                tree: Tree = tree
                trees.append(tree)
                specs.append(None)
                try:
                    specs[-1] = Specification(tree, factory).as_spec()
                except LookupError as e:
                    print(f"No specification found: {e}")
                except UnsupportedSpec as s:
                    print(f"Specification element not supported ({s})")
        except ValueError as e:
            print(f"Grammar: ({e})")
        if specs:
            # print("=" * 80)
            # print("Sentence:", sentence)
            # print("    Tags:", parser.tokenize(sentence).tags)
            # print("=" * 80)
            for tree, spec in zip(trees, specs):
                print(tree)
                print(spec)
            print()
            successful_sents += 1
        nspecs += len([spec for spec in specs if spec])
        ntrees += len(trees)
    return successful_sents, nspecs, ntrees, num_sents


def invoke_demo():
    """Demonstrates creation of invocations from specifically formatted strings, as well as usage."""

    invocation_triples = [
        ("print", "Prints {item:OBJ}", "Prints 0u32"),
        ("print", "Really prints {item:OBJ}", "Really prints \"HELLO WORLD\""),
        ("print", "Prints {item:OBJ} in {mod:ENUM}", "Really prints \"HELLO WORLD\""),
        ("print", "{item:OBJ} is printed", "`self.x()` is printed"),
        ("add", "{self:OBJ} is incremented by {rhs:IDENT}", "`self` is incremented by 1"),
        ("contains", "{self:OBJ} {MD} contain {item:OBJ}", "`self` must contain 0u32"),
        ("contains", "{self:OBJ} contains {item:OBJ}", "`self` contains 0u32"),
        ("contains", "contain {item:OBJ} {self:OBJ} {MD}", "contain 0u32, `self` will")
    ]

    invocations = [
        "Returns `true` if `self` is green.",
        "Returns `true` if `self` contains 0u32",
        "Returns the reciprocal of `self`"
    ]

    invoke_helper(invocations, invocation_triples)


def expr_demo():
    """Demonstrates expression parsing. Incomplete."""
    from pyrs_ast.expr import Expr

    print(Expr.from_str("u32::MIN + 1"))
    print(Expr.from_str("self.divide(rhs)"))
    print(Expr.from_str("0u32"))
    print(Expr.from_str("0xFFFu32"))
    print(Expr.from_str("1.2e3_f32"))
    print(Expr.from_str("1."))
    print(Expr.from_str("33"))
    print(Expr.from_str("hello"))

    # Parse expr, plug in types, and find corresponding fn


def search_demo():
    """Demonstrates searching for functions with multiple keywords."""

    ast = AstFile.from_path("../data/test3.rs")
    parser = Parser.default()
    # Query Demo
    words = [
        Phrase([Word("Remove", "VB", False, False)], parser),
        Phrase([Word("last", "JJ", False, True), Word("element", "NN", False, False)], parser),
        Phrase([Word("vector", "NN", True, False)], parser)
    ]
    print("Finding documentation matches.")
    items = ast.scope.find_fn_matches(Query(words))
    for item in items:
        print(item.sig_str())


def search_demo2():
    """Demonstrates searching for function arguments and phrases with synonyms."""
    ast = Crate.from_root_file("../data/test3.rs")
    parser = Parser.default()
    fields = [
        Phrase([Word("take", "VB", True, False)], parser),
        FnArg("usize")
    ]
    query = Query(fields, (Fn, Struct))
    for file in ast.files:
        for match in file.find_matches(query):
            print(match)

    print("FULL STDLIB")

    query = query_from_sentence("The minimum of two values", parser, (Fn, Struct))

    # doc_ast = DocCrate.from_root_dir(get_toolchains()[0])
    # for file in doc_ast.files:
    #     for match in file.find_matches(query):
    #         print(match)

    query = query_from_sentence("The smallest value", parser, (Fn, Struct, Const))

    # doc_ast = DocCrate.from_root_dir(get_toolchains()[0])
    for file in ast.files:
        for match in file.find_matches(query):
            print(match)


def query_formation_demo():
    """Demonstrates the creation of a query from a sentence, using the most relevant keywords."""
    print(
        [str(x) for x in query_from_sentence(
            """Removes and returns the element at position `index` within the vector""",
            Parser.default()
        ).fields]
    )

    print(
        [str(x) for x in query_from_sentence(
            """remove the last element from a vector and return it""",
            Parser.default()
        ).fields]
    )


def profiling(statement: str):
    import cProfile
    import pstats
    cProfile.run(statement, "stats")
    pstats.Stats("stats").sort_stats(pstats.SortKey.TIME).print_stats(20)


def tags_as_ents(doc: Doc):
    spans = []
    for i, token in enumerate(doc):
        span = doc[i: i + 1].char_span(0, len(token.text), label=token.tag_)
        spans.append(span)
    doc.set_ents(spans)


def render_ner(sentence: str, path: str, open_browser=False):
    from spacy import displacy
    from palette import ENTITY_COLORS
    import webbrowser

    sent = Parser.default().entities(sentence)

    html = displacy.render(
        sent["srl"][0],
        style="ent",
        options={"word_spacing": 30, "distance": 120, "colors": ENTITY_COLORS},
        page=True,
        manual=True,
    )

    with open(path, "w") as file:
        file.write(html)

    if open_browser:
        webbrowser.open(path)


def render_pos_tokens(sentence: str, path: str, idents=None, no_fix=False, open_browser=False):
    from spacy import displacy
    from palette import tag_color
    import webbrowser

    parser = Parser.default()
    if no_fix:
        sent = parser.tagger(sentence)
    else:
        sent = parser.tokenize(sentence, idents)
    tags_as_ents(sent.doc)
    colors = {tag: tag_color(tag) for tag in parser.tokens()}

    html = displacy.render(
        sent.doc,
        style="ent",
        options={"word_spacing": 30, "distance": 120, "colors": colors},
        page=True
    )
    with open(path, "w") as file:
        file.write(html)

    if open_browser:
        webbrowser.open(path)


def render_dep_graph(sentence: str, path: str, idents=None, no_fix=False, open_browser=False):
    from spacy import displacy
    from palette import tag_color
    import webbrowser

    parser = Parser.default()
    if no_fix:
        sent = parser.tagger(sentence)
    else:
        sent = parser.tokenize(sentence, idents)
    tags_as_ents(sent.doc)
    colors = {tag: tag_color(tag) for tag in parser.tokens()}

    html = displacy.render(
        sent.doc,
        style="dep",
        options={"word_spacing": 30, "distance": 140, "colors": colors},
        page=True
    )
    with open(path, "w") as file:
        file.write(html)

    if open_browser:
        webbrowser.open(path)


def render_parse_tree(sentence: str, path: str, idents=None, open_browser=False):
    from treevis import render_tree
    import webbrowser

    parser = Parser.default()
    tree = next(
        parser.parse_tree(sentence, idents=idents, attach_tags=False)
    )
    render_tree(tree, path)
    if open_browser:
        webbrowser.open(path)


def spec_from_sentence(sentence: str, idents=None):
    parser = Parser.default()
    tree = next(
        parser.parse_tree(sentence, idents=idents, attach_tags=False)
    )
    print(Specification(tree, None).as_spec())


def vis_demo():
    """Demonstrates entire pipeline from end to end."""
    ast = AstFile.from_path("../data/test3.rs")
    render_parse_tree(
        ast.scope.find_function("reciprocal").docs.sections()[0].sentences[0],
        idents={"self"},
        path="../images/reciprocal.pdf"
    )

    # sent = parser.tokenize_sentence(sentence, idents=idents)
    #
    # Path("../images/").mkdir(exist_ok=True)
    # svg = displacy.render(sent.doc, style="dep", options={"word_spacing": 30, "distance": 120})
    # output_path = Path("../images/pos_tags.svg")
    # output_path.open("w", encoding="utf-8").write(svg)

    render_pos_tokens("Returns `true`", idents=None, path="../images/pos_tags_pre2.html", no_fix=True)
    render_pos_tokens("Returns `true`", idents=None, path="../images/pos_tags_post2.html")


def run_on_rust_docs():
    from doc_json import get_toolchains, get_all_files
    files = get_all_files(get_toolchains()[0])

    sentences = []
    for item, _, _, _ in files:
        try:
            if isinstance(item, DocStruct):
                for method in item.methods:
                    if method is None:
                        continue
                    sentences += method.docs.sections()[0].sentences
            elif isinstance(item, DocFn):
                sentences += item.docs.sections()[0].sentences

        except IndexError:
            pass
    sentences = set(sentences)
    print(invoke_helper(sentences))


def invoke_testcases(path="base_grammar_test_cases.txt"):
    with open(path, "r") as file:
        invoke_helper([line.strip() for line in file.readlines()])


if __name__ == '__main__':
    formatter = logging.Formatter('[%(name)s/%(funcName)s] %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    logging.getLogger().setLevel(logging.INFO)

    # invoke_testcases()
    # search_demo2()
    # end_to_end_demo()
    run_on_rust_docs()
    # print(Parser.default().tokenize("Assigns 1 to `self.x`.").tags)
    # print(invoke_helper([
    #     "Assigns 1 to `self.x`.",
    #     "`self.x` is set to 1",
    #     "Sets `self.x` to 1",
    #     "1 is assigned to `self.x`",
    #     "1 is stored in `self.x`"
    # ], use_invokes=False))
    # print(invoke_helper([
    #     "Computes `self + rhs`, returning `None` if overflow occurred",
    #     "Returns `true` if and only if `self + rhs` overflows",
    #
    # ], use_invokes=False))

    # render_parse_tree("Computes `self + rhs`, returning `None` if overflow occurred.", "../images/overflow.pdf")
    # render_ner("Removes and returns the element at position index within the vector, shifting all elements after it to the left.", "/tmp/x.html", open_browser=True)
    # print([t.label_ for t in p.tokenize("Removes and returns the element at position index within the vector").doc.ents])
    # Motivate problems with what is being accomplished
    # problem and solution and reflection - therefore we do this
    # design writeup
    # bert library
    # search around

    # TODO: Detect duplicate invocations.
    # TODO: keyword in fn name, capitalization?
    # TODO: similarity metrics (capitalization, synonym distance via wordnet)
    # TODO: Decide spurious keywords

    # TODO: Mechanism to evaluate code
    # TODO: Add type to CODE item? eg. CODE_USIZE, CODE_BOOL, CODE_STR, then make CODE accept all of these
    #  std::any::type_name_of_val
    # TODO: allow specifying default value in #[invoke]
    #  eg. #[invoke(str, arg1 = 1usize, arg2 = ?, arg3 = ?)]
    # print(Struct.from_str("pub const unsafe extern \"rust-intrinsic\" fn size_of_val<T>(*const T) -> usize where T: ?Sized, {}"))
    # check Constructs a new, empty Vec<T>.
