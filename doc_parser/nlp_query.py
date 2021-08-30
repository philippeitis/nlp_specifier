from typing import List, Collection
import re
import itertools

from pyrs_ast.lib import Fn
from pyrs_ast.scope import Query, QueryField

from lemmatizer import is_synonym, lemmatize
from doc_parser import Parser


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
    def __init__(self, word: str, tag: str, allow_synonyms: bool, is_optional: bool, lemma: str = None):
        self.allow_synonyms = allow_synonyms
        self.word = word
        self.tag = tag
        self.is_optional = is_optional
        self._lemma = lemma

    def __str__(self):
        return self.word

    def matches_fn(self, fn):
        raise NotImplementedError()

    @property
    def lemma(self):
        if not self._lemma:
            self._lemma = lemmatize(self.word, self.tag)
        return self._lemma


class Phrase(QueryField):
    def __init__(self, phrase: List[Word], parser: Parser):
        self.parser = parser
        self.phrase = phrase
        regex_str = ""
        for word in self.phrase:
            val = get_regex_for_tag(re.escape(word.tag))
            if word.is_optional:
                regex_str += f"({val} )?"
            else:
                regex_str += f"{val} "

        regex_str = regex_str[:-1]
        self.tag_regex = re.compile(regex_str)
        self.regex_str = regex_str

    def matches(self, fn: Fn):
        """Determines whether the phrase matches the provided fn.
        To match the phrase, the function must contain all non-optional words, in sequence, with no gaps.
        Words do not need to have matching tenses or forms to be considered equal (using NLTK's lemmatizer).
        """

        def split_str(s: str, index: int):
            return s[:index], s[index:]

        docs = fn.docs.sections()
        if docs:
            sent = self.parser.tokenize(docs[0].body, idents=[ty.ident for ty in fn.inputs])
            s = " ".join(tag for tag in sent.tags)

            for match in self.tag_regex.finditer(s):
                prev, curr = split_str(s, match.start(0))
                curr, after = split_str(curr, match.end(0) - match.start(0))

                match_len = curr.count(" ") + 1
                match_start = prev.count(" ")

                match_tokens = sent.doc[match_start: match_start + match_len]
                word_iter = iter(match_tokens)
                matches = True
                for word in self.phrase:
                    match, word_iter = peek(word_iter)
                    if tags_similar(word.tag, match.tag_):
                        next(word_iter)
                        if word.lemma == match.lemma_:
                            continue
                        elif word.allow_synonyms:
                            if not is_synonym(word.word, match.text, word.tag):
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

    def __str__(self):
        return " ".join(str(x) for x in self.phrase)


def query_from_sentence(sentence, parser: Parser) -> Query:
    """Forms a query from a sentence.

    Stopwords (per Wordnet's stopwords), and words which are not verbs, nouns, adverbs, or adjectives, are all removed.
    Adverbs and adjectives are optional, and can be substituted with synonyms.
    """
    phrases = [[]]

    sent = parser.tokenize(sentence)

    for token in sent.doc:
        if token.is_stop:
            if phrases[-1]:
                phrases.append([])
        elif not is_one_of(token.tag_, {"RB", "VB", "NN", "JJ", "CODE", "LIT"}):
            if phrases[-1]:
                phrases.append([])
        else:
            is_describer = is_one_of(token.tag_, {"RB", "JJ"})
            phrases[-1].append(Word(token.text, token.tag_, allow_synonyms=is_describer, is_optional=is_describer, lemma=token.lemma_))
    return Query([Phrase(block, parser) for block in phrases if block])
