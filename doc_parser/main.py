from collections import defaultdict
from typing import List, Collection
import re
import logging
import itertools

import nltk
from nltk import Tree
from nltk.corpus import stopwords

from pyrs_ast.lib import LitAttr, Fn, HasItems
from pyrs_ast.scope import Query, FnArg, QueryField, Scope
from pyrs_ast import AstFile

from doc_parser import Parser, GRAMMAR_PATH, is_quote
from fn_calls import InvocationFactory, Invocation
from lemmatizer import lemma_eq, is_synonym
from grammar import Specification, generate_constructor_from_grammar

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


def peek(it):
    first = next(it)
    return first, itertools.chain([first], it)


def tags_similar(tag1: str, tag2: str) -> bool:
    if tag1 == tag2:
        return True
    if tag1.startswith("NN") and tag2.startswith("NN"):
        return True
    if tag1.startswith("VB") and tag2.startswith("VB"):
        return True
    return False


def is_one_of(tag: str, choices: Collection[str]) -> bool:
    return any(choice in tag for choice in choices)


def get_regex_for_tag(tag: str) -> str:
    if "VB" in tag:
        return "VB(D|G|N|P|Z)?"
    if "NN" in tag:
        return "NN(P|PS|S)?"
    if "RB" in tag:
        return "RB(R|S)?"
    if "JJ" in tag:
        return "JJ(R|S)?"
    return tag


class Word(QueryField):
    def __init__(self, word: str, allow_synonyms: bool, is_optional: bool):
        self.allow_synonyms = allow_synonyms
        self.word = word
        self.is_optional = is_optional

    def __str__(self):
        return self.word

    def matches_fn(self, fn):
        raise NotImplementedError()


class Phrase(QueryField):
    def __init__(self, phrase: List[Word], parser: Parser):
        self.parser = parser
        self.phrase = phrase
        s = " ".join(word.word for word in phrase)
        self.pos_tags, _ = parser.tokenize_sentence(s)
        regex_str = ""
        for word, tag in zip(self.phrase, self.pos_tags):
            val = get_regex_for_tag(re.escape(tag.value))
            if word.is_optional:
                regex_str += f"({val} )?"
            else:
                regex_str += f"{val} "

        regex_str = regex_str[:-1]
        self.tag_regex = re.compile(regex_str)
        self.regex_str = regex_str

    def __str__(self):
        return " ".join(str(x) for x in self.phrase)

    def matches(self, fn: Fn):
        """Determines whether the phrase matches the provided fn.
        To match the phrase, the function must contain all non-optional words, in sequence, with no gaps.
        Words do not need to have matching tenses or forms to be considered equal (using NLTK's lemmatizer).
        """

        def split_str(s: str, index: int):
            return s[:index], s[index:]

        docs = fn.docs.sections()
        if docs:
            pos_tags, words = self.parser.tokenize_sentence(docs[0].body, idents=[ty.ident for ty in fn.inputs])
            s = " ".join(tag.value for tag in pos_tags)

            for match in self.tag_regex.finditer(s):
                prev, curr = split_str(s, match.start(0))
                curr, after = split_str(curr, match.end(0) - match.start(0))

                match_len = curr.count(" ") + 1
                match_start = prev.count(" ")

                match_words = words[match_start: match_start + match_len]
                match_tags = pos_tags[match_start: match_start + match_len]
                word_iter = iter(zip(match_words, match_tags))
                matches = True
                for word, tag in zip(self.phrase, self.pos_tags):
                    match, word_iter = peek(word_iter)
                    m_word, m_tag = match
                    if tags_similar(tag.value, m_tag.value):
                        next(word_iter)
                        if lemma_eq(word.word, m_word, tag.value):
                            continue
                        elif word.allow_synonyms:
                            if not is_synonym(word.word, m_word, tag.value):
                                matches = False
                                break
                        else:
                            matches = False
                            break
                    elif word.is_optional:
                        continue
                    else:
                        matches = False
                        break
                if matches:
                    return True
        return False


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

    for child in tree:
        if tree_references_fn(fn, child):
            return True
    return False


def apply_specifications(fn: Fn, parser: Parser, scope: Scope, invoke_factory):
    """Creates a specification for the function, using all available sentences.
    Each sentence is cross-referenced with the grammar to form a syntax tree. If a syntax tree can be formed,
    this is treated as a valid specification, and is compiled into the Prusti annotation format.

    Prefers non-self-referential trees wherever possible.

    If no tree can be found for a particular sentence, no specification is added.
    """

    if not fn.should_specify():
        logging.info(f"fn {fn.ident} is not annotated with #[specify], and will not be specified.")
        return

    fn_idents = set(ty.ident for ty in fn.inputs)
    sections = fn.docs.sections()
    for section in sections:
        if section.header is not None:
            continue
        logging.info(f"Specifying documentation section {section.header} of {fn.ident}")
        for sentence in section.sentences:
            try:
                skipped = []
                parse_it = parser.parse_sentence(sentence, idents=fn_idents)
                spec = None
                for tree in parse_it:
                    if tree_references_fn(fn, tree):
                        logging.info("Skipping tree due to self-reference.")
                        skipped.append(tree)
                    else:
                        logging.info("Found tree without self-reference, applying.")
                        spec = Specification(tree, invoke_factory).as_spec()
                        break

                if spec is None:
                    spec = Specification(next(iter(skipped)), invoke_factory).as_spec()

                attr = LitAttr(spec)

                logging.info(f"[{sentence}] was transformed into the following specification: {attr}")
                fn.attrs.append(attr)
            except ValueError as v:
                logging.error(f"While specifying [{sentence}], error occurred: {v}. ")
                logging.info(
                    f"[{sentence}] has the following tags: {parser.tokenize_sentence(sentence, idents=fn_idents)[0]}"
                )
            except StopIteration as s:
                logging.info(f"No specification could be generated for [{sentence}]")
                logging.info(
                    f"[{sentence}] has the following tags: {parser.tokenize_sentence(sentence, idents=fn_idents)[0]}"
                )
                query = query_from_sentence(sentence, parser)
                logging.info("Found phrases: " + ", ".join(str(x) for x in query.fields))


def specify_item(item: HasItems, parser: Parser, scope: Scope, invoke_factory):
    for sub_item in item.items:
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
    logging.info(f"Searching for explicit invocations for fn {fn.ident}")

    for attr in fn.attrs:
        if str(attr.ident) == "invoke":
            invoke = str(attr.tokens[1].val.strip("\" "))
            logging.info(f"Found invocation [{invoke}] for fn {fn.ident}")
            invoke_factory.add_invocation(fn, Invocation.from_sentence(fn, invoke))

    for section in sections:
        if section.header is not None:
            continue
        logging.info(f"Determining descriptive sentence for {fn.ident}")
        descriptive_sentence = section.sentences[0]
        tokens, words = parser.tokenize_sentence(descriptive_sentence, idents=fn_idents)
        for sym, word in zip(tokens, words):
            if is_quote(word):
                continue

            word_replacements[word].add(sym.value)
            sym_replacements[sym.value].add(word)

        invoke_factory.add_fuzzy_invocation(fn, tokens, words)


def populate_grammar_helper(item: HasItems, parser: Parser, invoke_factory, word_replacements, sym_replacements):
    for sub_item in item.items:
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

    for word, syms in word_replacements.items():
        syms = " | ".join(f"\"{word}_{sym}\"" for sym in syms)
        grammar = grammar.replace(f"\"{word}\"", f"WD_{word}")
        replacement_grammar += f"WD_{word} -> {syms}\n"
    for sym, words in sym_replacements.items():
        syms = " | ".join(f"\"{word}_{sym}\"" for word in words)
        # grammar = grammar.replace(f"{sym} -> \"{sym}\"", "")
        replacement_grammar += f"{sym} -> {syms} | \"{sym}\"\n"

    with open(GRAMMAR_PATH) as f:
        full_grammar = f.read() + "\n# FUNCTION INVOCATIONS\n\n" + grammar + "\n\n# Word Replacements\n\n" + replacement_grammar

    return full_grammar, invoke_factory


def query_from_sentence(sentence, parser: Parser) -> Query:
    """Forms a query from a sentence.

    Stopwords (per Wordnet's stopwords), and words which are not verbs, nouns, adverbs, or adjectives, are all removed.
    Adverbs and adjectives are optional, and can be substituted with synonyms.
    """
    phrases = [[]]
    pos_tags, words = parser.tokenize_sentence(sentence)

    for pos_tag, word in zip(pos_tags, words):
        print(word, pos_tag)
        tag = pos_tag.value
        if word.lower() in STOPWORDS:
            if phrases[-1]:
                phrases.append([])
        elif not is_one_of(tag, {"RB", "VB", "NN", "JJ", "CODE", "LIT"}):
            if phrases[-1]:
                phrases.append([])
        else:
            is_describer = is_one_of(tag, {"RB", "JJ"})
            phrases[-1].append(Word(word, allow_synonyms=is_describer, is_optional=is_describer))
    return Query([Phrase(block, parser) for block in phrases if block])


def end_to_end_demo():
    """Demonstrates entire pipeline from end to end."""
    ast = AstFile.from_path("../data/test3.rs")
    grammar, invoke_factory = generate_grammar(ast)

    parser = Parser(grammar)
    print(grammar)
    specify_item(ast, parser, ast.scope, invoke_factory)
    print(ast)


def invoke_demo():
    """Demonstrates creation of invocations from specifically formatted strings, as well as usage."""

    from nltk import Tree
    from doc_parser import UnsupportedSpec

    class FnMock:
        def __init__(self, ident: str):
            self.ident = ident

    def populate_grammar_helper(sentences, parser, invoke_factory, word_replacements, sym_replacements):
        for sentence in sentences:
            if isinstance(sentence, tuple):
                invoke_factory.add_invocation(
                    FnMock(sentence[0]),
                    Invocation.from_sentence(FnMock(sentence[0]), sentence[1])
                )
                sentence = sentence[2]

            tokens, words = parser.tokenize_sentence(sentence)
            for sym, word in zip(tokens, words):
                if is_quote(word):
                    continue
                word_replacements[word].add(sym.value)
                sym_replacements[sym.value].add(word)

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

    grammar, factory = generate_grammar(invocation_triples + invocations, populate_grammar_helper)

    parser = Parser(grammar)

    sentences = invocations + [sentence for _, _, sentence in invocation_triples]
    for sentence in sentences:
        print("=" * 80)
        print("Sentence:", sentence)
        print("=" * 80)
        for tree in parser.parse_sentence(sentence):
            tree: Tree = tree
            try:
                print(Specification(tree, factory).as_spec())
            except LookupError as e:
                print(f"No specification found: {e}")
            except UnsupportedSpec as s:
                print(f"Specification element not supported ({s})")

            print(tree)
            print()


def expr_demo():
    """Demonstrates expression parsing. Incomplete."""

    import astx
    print(astx.parse_expr("self + rhs"))
    print(astx.parse_expr("self.divide(rhs)"))
    # Parse expr, plug in types, and find corresponding fn


def search_demo():
    """Demonstrates searching for functions with multiple keywords."""

    ast = AstFile.from_path("../data/test3.rs")
    parser = Parser.default()
    # Query Demo
    words = [
        Phrase([Word("Remove", False, False)], parser),
        Phrase([Word("last", False, True), Word("element", False, False)], parser),
        Phrase([Word("vector", True, False)], parser)
    ]
    print("Finding documentation matches.")
    items = ast.scope.find_fn_matches(Query(words))
    for item in items:
        print(item.sig_str())


def search_demo2():
    """Demonstrates searching for function arguments and phrases with synonyms."""
    ast = AstFile.from_path("../data/test2.rs")
    parser = Parser.default()

    # Query Demo
    words = [Word("Hello", False, False), Word("globe", True, False)]
    print("Finding documentation matches.")
    items = ast.scope.find_fn_matches(Query([Phrase(words, parser)]))
    for item in items:
        print(item.sig_str())
    print("Finding function argument matches")
    items = ast.scope.find_fn_matches(Query([FnArg(ast.scope.find_type("crate2::Lime"))]))
    for item in items:
        print(item.sig_str())


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


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    end_to_end_demo()

    # TODO: Detect duplicate invocations.
    # TODO: keyword in fn name, capitalization?
    # TODO: similarity metrics (capitalization, synonym distance via wordnet)
    # TODO: Decide spurious keywords

    # TODO: Mechanism to evaluate code
    # TODO: Add type to CODE item? eg. CODE_USIZE, CODE_BOOL, CODE_STR, then make CODE accept all of these

    # TODO: allow specifying default value in #[invoke]
    #  eg. #[invoke(str, arg1 = 1usize, arg2 = ?, arg3 = ?)]
