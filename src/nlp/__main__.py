import logging
from pathlib import Path
from sys import path

import click
import uvicorn
from spacy.tokens import Doc

path.append(str(Path(__file__).parent))

from tokenizer import Tokenizer


@click.group()
def cli():
    pass


def tags_as_ents(doc: Doc):
    spans = []
    for i, token in enumerate(doc):
        span = doc[i : i + 1].char_span(0, len(token.text), label=token.tag_)
        spans.append(span)
    doc.set_ents(spans)


@cli.group()
def render():
    """Visualization of various components in the system's pipeline."""
    pass


@render.command("dep")
@click.argument("sentence", nargs=1)
@click.option(
    "--open_browser", "-o", default=False, help="Opens file in browser", is_flag=True
)
@click.option(
    "--retokenize/--no-retokenize", "-r/-R", default=True, help="Applies retokenization"
)
@click.option(
    "--path", default=Path("./images/pos_tags.html"), help="Output path", type=Path
)
def render_dep_graph(sentence: str, open_browser: bool, retokenize: bool, path: Path):
    import webbrowser

    from spacy import displacy
    from visualization.palette import tag_color

    tokenizer = Tokenizer()
    if retokenize:
        sent = tokenizer.tagger(sentence)
    else:
        sent = tokenizer.tokenize(sentence)
    tags_as_ents(sent.doc)
    colors = {tag: tag_color(tag.tag_) for tag in sent.doc}

    html = displacy.render(
        sent.doc,
        style="dep",
        options={"word_spacing": 30, "distance": 140, "colors": colors},
        page=True,
    )

    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(html)

    if open_browser:
        webbrowser.open(str(path))


@render.command("pos")
@click.argument("sentence", nargs=1)
@click.option(
    "--open_browser", "-o", default=False, help="Opens file in browser", is_flag=True
)
@click.option(
    "--retokenize/--no-retokenize", "-r/-R", default=True, help="Applies retokenization"
)
@click.option(
    "--path", default=Path("./images/pos_tags.html"), help="Output path", type=Path
)
# @click.option('--idents', nargs=-1, help="Idents in string")
def render_pos(sentence: str, open_browser: bool, retokenize: bool, path: Path):
    """Renders the part of speech tags in the provided sentence."""
    import webbrowser

    from spacy import displacy
    from visualization.palette import tag_color

    tokenizer = Tokenizer()
    if retokenize:
        sent = tokenizer.tokenize(sentence)
    else:
        sent = tokenizer.tagger(sentence)

    tags_as_ents(sent.doc)
    colors = {tag: tag_color(tag) for tag in tokenizer.tokens()}

    html = displacy.render(
        sent.doc,
        style="ent",
        options={"word_spacing": 30, "distance": 120, "colors": colors},
        page=True,
    )

    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(html)

    if open_browser:
        webbrowser.open(str(path))


def render_entities(sentence: str, entity_type: str, open_browser: bool, path: Path):
    """Renders the NER or SRL entities in the provided sentence."""
    import webbrowser

    from spacy import displacy
    from visualization.palette import ENTITY_COLORS

    sent = Tokenizer().entities(sentence)
    entity_type = entity_type.lower()

    if entity_type == "ner":
        entities = sent[entity_type]
    else:
        entities = sent[entity_type][0]

    html = displacy.render(
        entities,
        style="ent",
        options={"word_spacing": 30, "distance": 120, "colors": ENTITY_COLORS},
        page=True,
        manual=True,
    )

    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(html)

    if open_browser:
        webbrowser.open(str(path))


@render.command("srl")
@click.argument("sentence", nargs=1)
@click.option(
    "--open_browser", "-o", default=False, help="Opens file in browser", is_flag=True
)
@click.option(
    "--path", default=Path("./images/srl.html"), help="Output path", type=Path
)
def render_srl(sentence: str, open_browser: bool, path: Path):
    """Renders the SRL entities in the provided sentence."""
    render_entities(sentence, "SRL", open_browser, path)


@render.command("ner")
@click.argument("sentence", nargs=1)
@click.option(
    "--open_browser", "-o", default=False, help="Opens file in browser", is_flag=True
)
@click.option(
    "--path", default=Path("./images/srl.html"), help="Output path", type=Path
)
def render_ner(sentence: str, open_browser: bool, path: Path):
    """Renders the NER entities in the provided sentence."""
    render_entities(sentence, "NER", open_browser, path)


@cli.command()
@click.option("--port", "-p", default=5000, help="Port to listen on")
@click.option("--host", default="0.0.0.0", help="Host address")
def launch(host: str, port: int):
    """Launches the server on the specified host, listening on the specified port."""
    from server import app, init_loggers

    init_loggers()
    uvicorn.run(app, host=host, port=port, log_level=logging.INFO)


if __name__ == "__main__":
    cli()
