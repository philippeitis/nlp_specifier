from tokenizer import Tokenizer, SpacyModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Space:
    def __init__(self, width: int):
        self.width = width

    def __str__(self):
        return " " * self.width


class NaiveSimilarity:
    """
    Uses the word-vector cosine similarity metric. In practice, is quite fast,
    but does not detect reordering of words.

    Additionally, words like "maximum" and "minimum" tend to have similar
    classifications (though more synonymous words will be more similar)
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sent1: str, sent2: str):
        s1 = self.tokenizer.tokenize(sent1)
        s2 = self.tokenizer.tokenize(sent2)
        return s1.doc.similarity(s2.doc)


class SimilarityFilter:
    """
    Uses the word-vector cosine similarity metric, applying some filter over the individual
    tokens (words). Refer to NaiveSimilarity docs for information on strengths and weaknesses
    """

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
    """
    Filters out all stop words from the sentence and computes the word-vector cosine similarity metric
    on the resulting sentence.
    """

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer, lambda t: not t.is_stop)


class SimilarityNouns(SimilarityFilter):
    """
    Filters out all words which are not nouns or verbs from the sentence and computes the word-vector cosine similarity
     metric on the resulting sentence.
    """
    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer, lambda t: t.pos_ in {'NOUN', 'PROPN', "VERB"})


class SimilarityBert:
    """
    Computes similarity using BERT, which is sensitive to the context in which words appear.

    Is very effective at detecting dissimilar sentences, but underperforms when measuring if sentences
    are similar.
    """
    def __init__(self):
        # Takes a long time to load these models
        model_name = "bert-base-cased-finetuned-mrpc"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def __call__(self, sent_a: str, sent_b: str):
        tokens = self.tokenizer(sent_a, sent_b, return_tensors="pt")
        classification_logits = self.model(**tokens)
        classification_logits = classification_logits[0]
        results = torch.softmax(classification_logits, dim=1).tolist()[0]
        return results[1]


def similarity_metrics(sentence_pairs):
    p = Tokenizer(model=SpacyModel.EN_LG)
    s_naive = NaiveSimilarity(p)
    s_nostop = SimilarityNoStop(p)
    s_nouns = SimilarityNouns(p)
    s_bert = SimilarityBert()

    sims = [
        ("naive", s_naive),
        ("nostop", s_nostop),
        ("nouns", s_nouns),
        ("bert", s_bert)
    ]

    for sent1, sent2 in sentence_pairs:
        print("=" * 80)
        print(f"{Space(4)}S1: {sent1}")
        print(f"{Space(4)}S2: {sent2}")

        for name, sim_metric in sims:
            print(f"{Space(8)}{name}: {sim_metric(sent1, sent2)}")


# TODO: Expose this in a server API
if __name__ == '__main__':
    sentence_pairs = [
        # Tests similarity metrics on opposing words
        # BERT is more sensitive to this replacement
        ("Returns the maximum of two `f32` values.", "Returns the minimum of two `f32` values."),
        # Tests similarity metrics on deletion
        # BERT is quite sensitive to this compared to other models
        ("Returns the minimum of two `f32` values.", "Returns the minimum of two values."),
        # Tests sensitivity to tense (again, outsized effect with BERT)
        ("Delete the last element of self.", "Remove the last element of self."),
        ("Deletes the last element of self.", "Removes the last element of self."),
        # Tests similarity metrics on different sentence
        # In BERT, this is very dissimilar, but by default, quite similar
        ("Delete the last element of self.", "Returns the maximum of two `f32` values."),
        ("The last element of self.", "self the last element of.")
    ]

    similarity_metrics(sentence_pairs)
