from typing import List, Collection
import re
import itertools

try:
    from tokenizer import Tokenizer
except ImportError:
    from nlp.tokenizer import Tokenizer


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
    if is_one_of(tag, {"CODE", "LIT"}):
        return "(CODE|LIT)"
    return tag


class Word:
    EVAL_COST = 1e-3

    def __init__(self, word: str, tag: str, allow_synonyms: bool, is_optional: bool, lemma: str = None):
        self.allow_synonyms = allow_synonyms
        self.word = word
        self.tag = tag
        self.is_optional = is_optional
        self.lemma = lemma

    def __str__(self):
        return self.word


class Phrase:
    EVAL_COST = 1.

    def __init__(self, phrase: List[Word], tokenizer: Tokenizer):
        self.tokenizer = tokenizer
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

    def matches(self, item):
        """Determines whether the phrase matches the provided fn.
        To match the phrase, the function must contain all non-optional words, in sequence, with no gaps.
        Words do not need to have matching tenses or forms to be considered equal (using NLTK's lemmatizer).
        """

        def split_str(s: str, index: int):
            return s[:index], s[index:]

        docs = item.docs.sections()
        if isinstance(item, Fn):
            idents = {ty.ident for ty in item.inputs}
        else:
            idents = None

        if docs:
            for sentx in docs[0].sentences:
                sent = self.tokenizer.tokenize(sentx, idents=idents)
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


def query_from_sentence(sentence, tokenizer: Tokenizer, *args, **kwargs) -> "Query":
    """Forms a query from a sentence.

    Stopwords (per Wordnet's stopwords), and words which are not verbs, nouns, adverbs, or adjectives, are all removed.
    Adverbs and adjectives are optional, and can be substituted with synonyms.
    """
    phrases = [[]]

    sent = tokenizer.tokenize(sentence)

    for token in sent.doc:
        if token.is_stop or token.tag_ in {"CODE", "LIT"}:
            if phrases[-1]:
                phrases.append([])
        elif not is_one_of(token.tag_, {"RB", "VB", "NN", "JJ"}):
            if phrases[-1]:
                phrases.append([])
        else:
            is_describer = is_one_of(token.tag_, {"RB", "JJ"})
            phrases[-1].append(
                Word(token.text, token.tag_, allow_synonyms=is_describer, is_optional=is_describer, lemma=token.lemma_))
    return Query([Phrase(block, tokenizer) for block in phrases if block], *args, **kwargs)


class SimPhrase:
    EVAL_COST = 0.8

    def __init__(self, phrase: str, tokenizer: Tokenizer, cutoff=0.85):
        self.tokenizer = tokenizer
        self.phrase = tokenizer.tokenize(phrase).doc
        self.cutoff = cutoff
        self.similarity_cache = {}
        self.sents_seen = 0

    def matches(self, item):
        """Determines whether the phrase matches the provided fn.
        To match the phrase, the function must contain at least one sentence which is sufficiently
         similar to the query string."""
        docs = item.docs.sections()
        if isinstance(item, Fn):
            idents = {ty.ident for ty in item.inputs}
        else:
            idents = None

        if docs:
            for sentx in docs[0].sentences:
                self.sents_seen += 1
                similarity = self.sent_similarity(sentx, idents)
                if similarity > self.cutoff:
                    self.similarity_cache[item] = similarity
                    return True
        return False

    def sent_similarity(self, sent, idents=None):
        """Determines whether the phrase matches the provided fn.
        To match the phrase, the function must contain at least one sentence which is sufficiently
         similar to the query string."""
        sent = self.tokenizer.tokenize(sent, idents=idents).doc
        similarity = sent.similarity(self.phrase)
        return similarity

    def any_similar(self, sents: List[str], cutoff, idents=None):
        """Determines whether the phrase matches the provided fn.
        To match the phrase, the function must contain at least one sentence which is sufficiently
         similar to the query string."""
        max_sim = -1.0
        for sent in sents:
            self.sents_seen += 1
            sent = self.tokenizer.tokenize(sent, idents=idents).doc
            similarity = sent.similarity(self.phrase)
            if similarity > cutoff:
                return True
            max_sim = max(max_sim, similarity)
        return False

    def __str__(self):
        return " ".join(str(x) for x in self.phrase)
