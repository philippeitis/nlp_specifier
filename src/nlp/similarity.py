from tokenizer import Tokenizer, SpacyModel, GRAMMAR_PATH


class NaiveSimilarity:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sent1: str, sent2: str):
        s1 = self.tokenizer.tokenize(sent1)
        s2 = self.tokenizer.tokenize(sent2)
        return s1.doc.similarity(s2.doc)


class SimilarityFilter:
    def __init__(self, tokenizer: Tokenizer, filter_fn):
        self.tokenizer = tokenizer
        self.filter_fn = filter_fn

    def filtered_sentence(self, sent):
        s = self.tokenizer.tokenize(sent)
        sent = " ".join(str(t) for t in s.doc if self.filter_fn(t))
        return self.tokenizer.tokenize(sent)

    def __call__(self, sent1: str, sent2: str):
        s1 = self.filtered_sentence(sent1)
        s2 = self.filtered_sentence(sent2)
        return s1.doc.similarity(s2.doc)


class SimilarityNoStop(SimilarityFilter):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer, lambda t: not t.is_stop)


class SimilarityNouns(SimilarityFilter):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer, lambda t: t.pos_ in {'NOUN', 'PROPN', "VERB"})


def transformers_similarity():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = "bert-base-cased-finetuned-mrpc"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    sequence_0 = "Returns the maximum of two `f32` values."
    sequence_1 = "Returns the minimum of two values."

    tokens = tokenizer(sequence_0, sequence_1, return_tensors="pt")
    print(tokens)
    print(tokenizer(sequence_0, return_tensors="pt"))
    classification_logits = model(**tokens)
    print(classification_logits)
    classification_logits = classification_logits[0]
    results = torch.softmax(classification_logits, dim=1).tolist()[0]
    print(torch.softmax(classification_logits, dim=1))
    print(results[1], sum(results))


def spacy_similarity():
    p = Tokenizer.from_path(GRAMMAR_PATH, model=SpacyModel.EN_LG)
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
        # (fn_rm, fn_pop),
        # (fn_rm, fn_contains),
        # (fn_pop, fn_contains),

    ]
    for sent1, sent2 in sents:
        print("=" * 80)
        print(f"    S1: {sent1}")
        print(f"    S2: {sent2}")

        for name, sim_metric in sims:
            print(f"{name}: {sim_metric(sent1, sent2)}")

    s1 = "Delete the last element of self."
    s2 = "Remove the last element of self."
    s1d = p.tokenize(s1).doc
    s2d = p.tokenize(s2).doc
    print(s1d.similarity(s2d))
    print(s1d[0].similarity(s2d[0]))


if __name__ == '__main__':
    transformers_similarity()
