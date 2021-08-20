from nltk.corpus.reader import NOUN, VERB, ADV, ADJ
from nltk.stem import WordNetLemmatizer

__all__ = ["lemmatize", "lemma_eq"]
LEMMATIZER = WordNetLemmatizer()


def nltk_pos(tag: str):
    if tag in {VERB, NOUN, ADV, ADJ}:
        return tag
    if "VB" in tag:
        return VERB
    if "NN" in tag or tag in {"CODE", "LIT"}:
        return NOUN
    if "RB" in tag:
        return ADV
    if "JJ" in tag:
        return ADJ
    return NOUN


def lemmatize(word: str, pos: str = NOUN) -> str:
    return LEMMATIZER.lemmatize(word, nltk_pos(pos))


def lemma_eq(word1: str, word2: str, pos: str = NOUN) -> bool:
    return lemmatize(word1.lower(), pos) == lemmatize(word2.lower(), pos)
