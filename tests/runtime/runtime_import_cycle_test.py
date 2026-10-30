"""Keeps midrun.runtime.runtime out of every import cycle.

REST, FUSE and the virtual files read the runtime, so the runtime must not import
them back. Every import counts, also one under TYPE_CHECKING or inside a function:
those only hide a cycle from the interpreter, not from a reader. AST-based, so the
module names in docstrings do not count.
"""

import ast
from pathlib import Path

import midrun

_RUNTIME = "midrun.runtime.runtime"


def _module_paths() -> dict[str, Path]:
    root = Path(midrun.__file__).parent.parent
    paths: dict[str, Path] = {}
    for path in Path(midrun.__file__).parent.rglob("*.py"):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        paths[".".join(parts)] = path
    return paths


def _imported_modules(path: Path, known: set[str]) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules & known


def test_the_runtime_module_is_in_no_import_cycle() -> None:
    paths = _module_paths()
    imports = {
        name: _imported_modules(path, set(paths)) for name, path in paths.items()
    }
    # Each module that the runtime reaches, with the chain of imports to it.
    chains: dict[str, list[str]] = {_RUNTIME: [_RUNTIME]}
    pending = [_RUNTIME]
    while pending:
        module = pending.pop()
        for imported in sorted(imports[module] - chains.keys()):
            chains[imported] = [*chains[module], imported]
            pending.append(imported)

    cycles = [
        " -> ".join([*chain, _RUNTIME])
        for module, chain in chains.items()
        if module != _RUNTIME and _RUNTIME in imports[module]
    ]

    assert not cycles, f"{_RUNTIME} imports modules that import it back: {cycles}"
