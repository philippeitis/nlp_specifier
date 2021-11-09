import stanza
from stanza.server import CoreNLPClient


class Tree:
    def __init__(self, parse_tree):
        self.value = parse_tree.value
        self.score = parse_tree.score
        self.children = [Tree(child) for child in parse_tree.child]

    def __str__(self):
        if self.children:
            return f"({self.value} {' '.join(str(child) for child in self.children)})"
        return f"{self.value}"

if __name__ == '__main__':
    stanza.install_corenlp()
    stanza.download()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

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
            print(Tree(ann.sentence[0].parseTree.child[0]))
            print(nlp(line).sentences[0].constituency)
