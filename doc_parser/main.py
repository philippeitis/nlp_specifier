from typing import List
import re
import logging

import itertools
from nltk.corpus import wordnet

from pyrs_ast.lib import LitAttr, Fn, HasItems
from pyrs_ast.scope import Query, FnArg, QueryField, Scope
from pyrs_ast import AstFile, print_ast

from doc_parser import Parser, Specification, GRAMMAR_PATH, generate_constructor_from_grammar, is_quote
from fn_calls import InvocationFactory



def peek(it):
    first = next(it)
    return first, itertools.chain([first], it)


def is_synonym(word1: str, word2: str) -> bool:
    word1 = word1.lower()
    word2 = word2.lower()
    for syn in wordnet.synsets(word1):
        for lemma in syn.lemma_names():
            if lemma == word2 and lemma != word1:
                return True
    return False


class Word(QueryField):
    def __init__(self, word: str, synonyms: bool, optional: bool):
        self.synonyms = synonyms
        self.word = word
        self.optional = optional

    def matches_fn(self, fn):
        return


class Phrase(QueryField):
    def __init__(self, phrase: List[Word], parser: Parser):
        self.parser = parser
        self.phrase = phrase
        s = " ".join(word.word for word in phrase)
        self.pos_tags = parser.tokenize_sentence(s)[0]
        regex_str = ""
        for word, tag in zip(self.phrase, self.pos_tags):
            val = re.escape(tag.value)
            if word.optional:
                regex_str += f"({val})? "
            else:
                regex_str += f"{val} "
        regex_str = regex_str[:-1]
        self.tag_regex = re.compile(regex_str)
        self.regex_str = regex_str

    def matches(self, fn):
        def split_str(s: str, index: int):
            return s[:index], s[index:]

        docs = fn.docs.sections()
        if docs:
            pos_tags, words = self.parser.tokenize_sentence(docs[0].body, idents=[ty.ident for ty in fn.inputs])
            s = " ".join(tag.value for tag in pos_tags)

            for match in self.tag_regex.finditer(s):
                prev, curr = split_str(s, match.start(0))
                curr, after = split_str(curr, match.end(0) - match.start(0))
                match_len = len(curr.split(" "))
                match_start = len(prev.split(" ")) - 1
                match_words = words[match_start: match_start + match_len]
                match_tags = pos_tags[match_start: match_start + match_len]

                word_iter = iter(zip(match_words, match_tags))
                matches = True
                for word, tag in zip(self.phrase, self.pos_tags):
                    match, word_iter = peek(word_iter)
                    m_word, m_tag = match
                    if m_tag.value == tag.value:
                        next(word_iter)
                        if word.word.lower() == m_word.lower():
                            continue
                        elif word.synonyms:
                            if not is_synonym(word.word, m_word):
                                matches = False
                                break
                        else:
                            matches = False
                            break
                    elif word.optional:
                        continue
                    else:
                        matches = False
                        break
                if matches:
                    return True

        return False


def apply_specifications(fn: Fn, parser: Parser, scope: Scope, invoke_factory):
    idents = [type_tuple[1] for type_tuple in fn.type_tuples()]
    sections = fn.docs.sections()
    for section in sections:
        if section.header is not None:
            continue
        logging.info(f"Specifying documentation section {section.header} of {fn.ident}")
        for sentence in section.sentences:
            try:
                spec = Specification(next(parser.parse_sentence(sentence, idents=idents)), invoke_factory).as_spec()
                attr = LitAttr(spec)

                logging.info(f"[{sentence}] was transformed into the following specification: {attr}")
                fn.attrs.append(attr)
            except ValueError as v:
                logging.error(f"While specifying [{sentence}], error occurred: {v}")
            except StopIteration as s:
                logging.info(f"No specification could be generated for [{sentence}]")


def specify_item(item: HasItems, parser: Parser, scope: Scope, invoke_factory):
    for sub_item in item.items:
        if isinstance(sub_item, Fn):
            apply_specifications(sub_item, parser, scope, invoke_factory)
        elif isinstance(sub_item, HasItems):
            specify_item(sub_item, parser, scope, invoke_factory)


def find_specifying_sentence(fn: Fn, parser: Parser, invoke_factory: InvocationFactory, word_replacements,
                             sym_replacements):
    sections = fn.docs.sections()
    for section in sections:
        if section.header is not None:
            continue
        logging.info(f"Determining descriptive sentence for {fn.ident}")
        descriptive_sentence = section.sentences[0]
        tokens, words = parser.tokenize_sentence(descriptive_sentence)
        for sym, word in zip(tokens, words):
            if is_quote(word):
                continue
            if word not in word_replacements:
                word_replacements[word] = set()
            if sym.value not in sym_replacements:
                sym_replacements[sym.value] = set()
            word_replacements[word].add(sym.value)
            sym_replacements[sym.value].add(word)
        invoke_factory.add_invocation(fn.ident, descriptive_sentence)


def populate_grammar_helper(item: HasItems, parser: Parser, invoke_factory, word_replacements, sym_replacements):
    for sub_item in item.items:
        if isinstance(sub_item, Fn):
            find_specifying_sentence(sub_item, parser, invoke_factory, word_replacements, sym_replacements)
        elif isinstance(sub_item, HasItems):
            populate_grammar_helper(sub_item, parser, invoke_factory, word_replacements, sym_replacements)


def generate_grammar(ast: AstFile):
    word_replacements = {}
    sym_replacements = {}
    invoke_factory = InvocationFactory(generate_constructor_from_grammar)
    populate_grammar_helper(ast, Parser.default(), invoke_factory, word_replacements, sym_replacements)

    grammar = invoke_factory.grammar()
    replacement_grammar = ""
    for word, syms in word_replacements.items():
        syms = " | ".join(f"\"{word}_{sym}\"" for sym in syms)
        grammar = grammar.replace(f"\"{word}\"", f"WD_{word}")
        replacement_grammar += f"WD_{word} -> {syms}\n"
    for sym, words in sym_replacements.items():
        syms = " | ".join(f"\"{word}_{sym}\"" for word in words)
        grammar = grammar.replace(f"{sym} -> \"{sym}\"", "")
        replacement_grammar += f"{sym} -> {syms} | \"{sym}\"\n"
    with open(GRAMMAR_PATH) as f:
        full_grammar = f.read() + "\n" + grammar + "\n" + replacement_grammar

    return full_grammar, invoke_factory

# TODO: keyword (in doc or in fn name, structural / phrase search, synonyms ok? capitalization ok?
# TODO: similarity metrics

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    ast = AstFile.from_path("../data/test3.rs")

    grammar, invoke_factory = generate_grammar(ast)
    parser = Parser(grammar)

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

    specify_item(ast, parser, ast.scope, invoke_factory)
    print_ast(ast)

    print(grammar)