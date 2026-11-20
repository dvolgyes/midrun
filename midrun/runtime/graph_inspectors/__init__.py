"""Architecture and autograd graph capture via torchview/torchviz.

Both Python packages are installed with Midrun.
SVG rendering also needs the Graphviz `dot` executable.
Rendering writes to a real file path (`graphviz.Digraph.render`), so both captures use
the tempdir-then-read-bytes technique to hand the result to `ArtifactStore` unchanged.

`torchview_dot` is the exception: `graphviz.Digraph.source` is built in pure Python and
never invokes the `dot` binary, so it needs only the `torchview` package. It backs the
live FUSE graph file (see `midrun.fuse.filesystem`), which favors that over rendering.
"""
