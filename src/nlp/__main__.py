import logging
import webbrowser
from pathlib import Path
from sys import path

import click
import uvicorn

path.append(str(Path(__file__).parent))


def render_outputs(html, open_browser: bool, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(html)

    if open_browser:
        webbrowser.open(str(path))


@click.group()
def cli():
    pass


@cli.group()
def render():
    """Visualization of various components in the system's pipeline."""
    pass


@render.command("deps")
@click.argument("sentence", nargs=1)
@click.option(
    "--retokenize/--no-retokenize", "-r/-R", default=True, help="Applies retokenization"
)
@click.option(
    "--path", default=Path("./images/pos_tags.html"), help="Output path", type=Path
)
@click.option(
    "--open_browser", "-o", default=False, help="Opens file in browser", is_flag=True
)
def render_dep_graph(sentence: str, retokenize: bool, open_browser: bool, path: Path):
    from visualization import render_dep_graph

    render_outputs(render_dep_graph(sentence, retokenize), open_browser, path)


@render.command("pos")
@click.argument("sentence", nargs=1)
@click.option(
    "--retokenize/--no-retokenize", "-r/-R", default=True, help="Applies retokenization"
)
@click.option(
    "--path", default=Path("./images/pos_tags.html"), help="Output path", type=Path
)
@click.option(
    "--open_browser", "-o", default=False, help="Opens file in browser", is_flag=True
)
# @click.option('--idents', nargs=-1, help="Idents in string")
def render_pos(sentence: str, retokenize: bool, open_browser: bool, path: Path):
    """Renders the part of speech tags in the provided sentence."""
    from visualization import render_pos

    render_outputs(render_pos(sentence, retokenize), open_browser, path)


def render_entities(sentence: str, entity_type: str, open_browser: bool, path: Path):
    """Renders the NER or SRL entities in the provided sentence."""
    from visualization import render_entities

    render_outputs(render_entities(sentence, entity_type), open_browser, path)


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
