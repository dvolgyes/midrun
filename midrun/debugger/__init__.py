"""Standalone, out-of-process tooling that attaches to a running Midrun-managed training process.

Talks to the target over its REST API and OS-level process introspection
(py-spy, gdb). Never imported by `midrun.mixin`/`midrun.runtime`/`midrun.fuse`
-- see `tests/debugger/import_boundary_test.py`.
"""
