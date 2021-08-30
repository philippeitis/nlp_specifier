from doc_parser import Parser
from pyrs_ast import AstFile


class NaiveSimilarity:
    def __init__(self, parser: Parser):
        self.parser = parser

    def __call__(self, sent1: str, sent2: str):
        s1 = self.parser.tokenize(sent1)
        s2 = self.parser.tokenize(sent2)
        return s1.doc.similarity(s2.doc)


class SimilarityFilter:
    def __init__(self, parser: Parser, filter_fn):
        self.parser = parser
        self.filter_fn = filter_fn

    def filtered_sentence(self, sent):
        s = self.parser.tokenize(sent)
        sent = " ".join(str(t) for t in s.doc if self.filter_fn(t))
        return self.parser.tokenize(sent)

    def __call__(self, sent1: str, sent2: str):
        s1 = self.filtered_sentence(sent1)
        s2 = self.filtered_sentence(sent2)
        return s1.doc.similarity(s2.doc)


class SimilarityNoStop(SimilarityFilter):
    def __init__(self, parser: Parser):
        super().__init__(parser, lambda t: not t.is_stop)


class SimilarityNouns(SimilarityFilter):
    def __init__(self, parser: Parser):
        super().__init__(parser, lambda t: t.pos_ in {'NOUN', 'PROPN', "VERB"})


if __name__ == '__main__':
    a = AstFile.from_path("../data/test5.rs")
    ty = a.scope.find_type("PVec")
    fn_rm = ty.methods[0].docs.sections()[0].sentences[0]
    fn_pop = ty.methods[1].docs.sections()[0].sentences[0]
    fn_contains = ty.methods[2].docs.sections()[0].sentences[0]

    p = Parser.default()
    s_naive = NaiveSimilarity(p)
    s_nostop = SimilarityNoStop(p)
    s_nouns = SimilarityNouns(p)

    sims = [
        ("naive", s_naive),
        ("nostop", s_nostop),
        ("nouns", s_nouns),
    ]

    sents = [
        ("hello world", "hello_globe"),
        (fn_rm, fn_pop),
        (fn_rm, fn_contains),
        (fn_pop, fn_contains),

    ]
    for sent1, sent2 in sents:
        print("=" * 80)
        print(f"    S1: {sent1}")
        print(f"    S2: {sent2}")

        for name, sim_metric in sims:
            print(f"{name}: {sim_metric(sent1, sent2)}")
