from enum import Enum
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from spacy import displacy
from spacy.tokens import Doc

from tokenizer import Sentence, Tokenizer

from .palette import ENTITY_COLORS, tag_color

router = APIRouter()


class Entity(str, Enum):
    NER = "ner"
    SRL = "srl"


def tags_as_ents(doc: Doc):
    spans = []
    for i, token in enumerate(doc):
        span = doc[i : i + 1].char_span(0, len(token.text), label=token.tag_)
        spans.append(span)
    doc.set_ents(spans)


@router.get("/render/pos", response_class=HTMLResponse)
def render_pos(sentence: str, retokenize: Optional[bool] = True):
    """Renders the part of speech tags in the provided sentence."""
    from .palette import tag_color

    tokenizer = Tokenizer()
    if retokenize:
        sent = tokenizer.tokenize(sentence)
    else:
        sent = Sentence(tokenizer.tagger(sentence, disable=["doc_tokens"]))

    tags_as_ents(sent.doc)
    colors = {tag: tag_color(tag) for tag, _, _ in sent.metadata}

    return displacy.render(
        sent.doc,
        style="ent",
        options={"word_spacing": 30, "distance": 120, "colors": colors},
        page=True,
    )


@router.get("/render/deps", response_class=HTMLResponse)
def render_dep_graph(sentence: str, retokenize: Optional[bool] = True):
    tokenizer = Tokenizer()
    if retokenize:
        sent = tokenizer.tokenize(sentence)
    else:
        sent = Sentence(tokenizer.tagger(sentence, disable=["doc_tokens"]))

    tags_as_ents(sent.doc)
    colors = {tag: tag_color(tag.tag_) for tag in sent.doc}

    return displacy.render(
        sent.doc,
        style="dep",
        options={"word_spacing": 30, "distance": 140, "colors": colors},
        page=True,
    )


@router.get("/render/{entity_type}", response_class=HTMLResponse)
def render_entities(sentence: str, entity_type: Entity):
    sent = Tokenizer().entities(sentence)
    entity_type = entity_type.lower()

    if entity_type == "ner":
        entities = sent[entity_type.value]
    else:
        entities = sent[entity_type.value][0]

    return displacy.render(
        entities,
        style="ent",
        options={"word_spacing": 30, "distance": 120, "colors": ENTITY_COLORS},
        page=True,
        manual=True,
    )
