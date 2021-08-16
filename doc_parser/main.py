from typing import List
import re

import itertools
from nltk.corpus import wordnet

from pyrs_ast.lib import LitAttr
from pyrs_ast.scope import Query, FnArg, QueryField
from pyrs_ast import AstFile

from doc_parser import Parser, Specification


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


def apply_specifications(self, parser: Parser, verbose: bool = False):
    def vprint(*args):
        if verbose:
            print(*args)

    idents = [type_tuple[1] for type_tuple in self.type_tuples()]

    sections = self.docs.sections()
    vprint([(section.header, section.lines, section.body) for section in self.docs.sections()])
    for section in sections:
        if section.header is not None:
            continue
        vprint("SECTION:", section.header, section.sentences)
        for sentence in section.sentences:
            try:
                attr = LitAttr(Specification(next(parser.parse_sentence(sentence, idents=idents))).as_spec())

                vprint("PRINTING SPEC:", attr)
                self.attrs.append(attr)
            except ValueError as v:
                vprint(f"Unexpected spec {v}")
            except StopIteration as s:
                vprint(f"Did not find spec for \"{sentence}\"")


# TODO: keyword (in doc or in fn name, structural / phrase search, synonyms ok? capitalization ok?
# TODO: similarity metrics

if __name__ == '__main__':
    ast = AstFile.from_path("../data/test2.rs")
    words = [Word("Hello", False, False), Word("globe", True, False)]
    print("Finding documentation matches.")
    parser = Parser.from_path()
    items = ast.scope.find_fn_matches(Query([Phrase(words, parser)]))
    for item in items:
        print(item)
    print("Finding function argument matches")
    items = ast.scope.find_fn_matches(Query([FnArg(ast.scope.find_type("crate2::Lime"))]))
    for item in items:
        print(item)
