"""Renders a Graphviz graph to SVG and DOT."""

import tempfile
from pathlib import Path
from typing import Any


def render_graph(visual_graph: Any) -> dict[str, str | bytes]:
    """Renders a `graphviz` graph and returns its SVG bytes and DOT source.

    Returns:
        `{"graph.svg": <svg bytes>, "graph.dot": <dot source>}`.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        visual_graph.render(
            filename="graph", directory=tmp_dir, format="svg", cleanup=True
        )
        svg_bytes = (Path(tmp_dir) / "graph.svg").read_bytes()
    return {"graph.svg": svg_bytes, "graph.dot": visual_graph.source}
