from spacy import displacy
from spacy.tokens import Doc

from ..tokenizer import Tokenizer
from .palette import ENTITY_COLORS, tag_color


def tags_as_ents(doc: Doc):
    spans = []
    for i, token in enumerate(doc):
        span = doc[i : i + 1].char_span(0, len(token.text), label=token.tag_)
        spans.append(span)
    doc.set_ents(spans)


def render_pos(sentence: str, retokenize: bool):
    """Renders the part of speech tags in the provided sentence."""
    from .palette import tag_color

    tokenizer = Tokenizer()
    if retokenize:
        sent = tokenizer.tokenize(sentence)
    else:
        sent = tokenizer.tagger(sentence)

    tags_as_ents(sent.doc)
    colors = {tag: tag_color(tag) for tag, _, _ in sent.metadata}

    return displacy.render(
        sent.doc,
        style="ent",
        options={"word_spacing": 30, "distance": 120, "colors": colors},
        page=True,
    )


def render_entities(sentence: str, entity_type: str):
    sent = Tokenizer().entities(sentence)
    entity_type = entity_type.lower()

    if entity_type == "ner":
        entities = sent[entity_type]
    else:
        entities = sent[entity_type][0]

    return displacy.render(
        entities,
        style="ent",
        options={"word_spacing": 30, "distance": 120, "colors": ENTITY_COLORS},
        page=True,
        manual=True,
    )


def render_dep_graph(sentence: str, retokenize: bool):
    tokenizer = Tokenizer()
    if retokenize:
        sent = tokenizer.tagger(sentence)
    else:
        sent = tokenizer.tokenize(sentence)
    tags_as_ents(sent.doc)
    colors = {tag: tag_color(tag.tag_) for tag in sent.doc}

    return displacy.render(
        sent.doc,
        style="dep",
        options={"word_spacing": 30, "distance": 140, "colors": colors},
        page=True,
    )
