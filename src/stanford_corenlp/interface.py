import stanza
from stanza.server import CoreNLPClient

if __name__ == '__main__':
    stanza.install_corenlp()
    with CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'],
            timeout=30000,
            memory='6G') as client:
        for line in [
            "will do something If x is true",
            "When x is true, does something",
            "He does something when x is true",
        ]:
            ann = client.annotate(line)
            print(ann.sentence[0].parseTree)
            print(ann.sentence[0].parseTree.child)