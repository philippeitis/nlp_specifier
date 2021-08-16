import itertools
from typing import Dict, Union, List
import re

from nltk.corpus import wordnet

from pyrs_ast.ast_types import Type


def is_synonym(word1: str, word2: str) -> bool:
    word1 = word1.lower()
    word2 = word2.lower()
    for syn in wordnet.synsets(word1):
        for lemma in syn.lemma_names():
            if lemma == word2 and lemma != word1:
                return True
    return False


def peek(it):
    first = next(it)
    return first, itertools.chain([first], it)


# Acts like a type factory, ensuring that only one instance of a type exists for a particular declaration.
class Scope:
    def __init__(self):
        self.named_types = {}
        self.structs = {}
        self.functions: Dict[str, "Fn"] = {}

        self.parent = None
        self.visibility = None

    def add_struct(self, name: str, struct):
        self.structs[name] = struct
        if name in self.named_types:
            self.named_types[name].register_struct(struct)

    def add_fn(self, name: str, fn):
        self.functions[name] = fn

    def find_function(self, fn: str):
        return self.functions.get(fn)

    def find_type(self, ty: str):
        return self.named_types.get(ty, self.structs.get(ty))

    def define_type(self, **kwargs) -> Type:
        ty = Type(**kwargs)
        name = ty.name()

        if name is None:
            return ty

        name = str(name)

        if name in self.named_types:
            return self.named_types[name]
        if name in self.structs:
            ty.register_struct(self.structs[name])
        self.named_types[name] = ty
        return ty

    def find_fn_matches(self, query: "Query"):
        """Return all functions and methods which match the particular set of queries, in no particular order."""
        res = set()
        for ty in self.structs.values():
            for method in ty.methods:
                if query.matches_fn(method):
                    res.add(method)
        for ty in self.named_types.values():
            for method in ty.methods:
                if query.matches_fn(method):
                    res.add(method)
        for fn in self.functions.values():
            if query.matches_fn(fn):
                res.add(fn)

        return res


class FnArg:
    def __init__(self, xtype, position: int = None, is_input: bool = True):
        self.type = xtype
        self.position = position
        self.is_input = is_input

    def matches(self, fn):
        # TODO: Handle tuple types.
        items = fn.inputs if self.is_input else fn.output
        if self.position is not None:
            if len(items) <= self.position:
                return False
            return self.type == items[self.position].ty
        else:
            return any(self.type == item.ty for item in items)


class Word:
    def __init__(self, word: str, synonyms: bool, optional: bool):
        self.synonyms = synonyms
        self.word = word
        self.optional = optional

    def matches_fn(self, fn):
        return


class Phrase:
    def __init__(self, phrase: List[Word]):
        from lib import get_parser
        parser = get_parser()

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

        from lib import get_parser
        parser = get_parser()
        docs = fn.extract_docs().sections()
        if docs:
            pos_tags, words = parser.tokenize_sentence(docs[0].body, idents=[ty.ident for ty in fn.inputs])
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


class Query:
    def __init__(self, fields):
        self.fields = fields

    def matches_fn(self, fn):
        return all(field.matches(fn) for field in self.fields)

# Todo: keyword (in doc or in fn name, structural / phrase search, synonyms ok? capitalization ok?
# TODO: similarity metrics
