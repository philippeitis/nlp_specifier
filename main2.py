import spacy
from spacy.symbols import ORTH
from spacy.tokens import Doc

if __name__ == '__main__':
    nlp = spacy.blank("en")
    words = ["Hello", ",", "world", "!"]
    spaces = [False, False, False, False]
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    print(doc.text)
    print([(t.text, t.text_with_ws, t.whitespace_) for t in doc])