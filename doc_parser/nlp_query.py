from typing import List, Collection
import re
import itertools

import nltk
from nltk.corpus import stopwords

from pyrs_ast.lib import Fn
from pyrs_ast.scope import Query, QueryField

from lemmatizer import lemma_eq, is_synonym
from doc_parser import Parser

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
    def __init__(self, word: str, tag: str, allow_synonyms: bool, is_optional: bool):
        self.allow_synonyms = allow_synonyms
        self.word = word
        self.tag = tag
        self.is_optional = is_optional

    def __str__(self):
        return self.word

    def matches_fn(self, fn):
        raise NotImplementedError()


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
            sent = self.parser.tokenize_sentence(docs[0].body, idents=[ty.ident for ty in fn.inputs])
            s = " ".join(tag for tag in sent.tags)

            for match in self.tag_regex.finditer(s):
                prev, curr = split_str(s, match.start(0))
                curr, after = split_str(curr, match.end(0) - match.start(0))

                match_len = curr.count(" ") + 1
                match_start = prev.count(" ")

                match_words = sent.words[match_start: match_start + match_len]
                match_tags = sent.tags[match_start: match_start + match_len]
                word_iter = iter(zip(match_words, match_tags))
                matches = True
                for word in self.phrase:
                    match, word_iter = peek(word_iter)
                    m_word, m_tag = match
                    if tags_similar(word.tag, m_tag):
                        next(word_iter)
                        if lemma_eq(word.word, m_word, word.tag):
                            continue
                        elif word.allow_synonyms:
                            if not is_synonym(word.word, m_word, word.tag):
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

    sent = parser.tokenize_sentence(sentence)

    for tag, word in zip(sent.tags, sent.words):
        if word.lower() in STOPWORDS:
            if phrases[-1]:
                phrases.append([])
        elif not is_one_of(tag, {"RB", "VB", "NN", "JJ", "CODE", "LIT"}):
            if phrases[-1]:
                phrases.append([])
        else:
            is_describer = is_one_of(tag, {"RB", "JJ"})
            phrases[-1].append(Word(word, tag, allow_synonyms=is_describer, is_optional=is_describer))
    return Query([Phrase(block, parser) for block in phrases if block])
