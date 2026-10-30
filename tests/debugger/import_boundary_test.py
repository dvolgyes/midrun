"""Keeps midrun.debugger out-of-process.

It must never import midrun.runtime, midrun.mixin, or midrun.fuse -- it may
only talk to a target process over REST (`hold_client`) and OS-level attach
(py-spy/gdb against a PID). AST-based, not text grep, so prose mentions of these
module names in docstrings (there are several, describing why the boundary
exists) don't produce false positives.
"""

import ast
from pathlib import Path

import midrun.debugger

_FORBIDDEN_PREFIXES = ("midrun.runtime", "midrun.mixin", "midrun.fuse")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in _FORBIDDEN_PREFIXES
    )


def test_debugger_package_has_no_trainer_side_imports() -> None:
    package_dir = Path(midrun.debugger.__file__).parent
    violations = [
        (path, module)
        for path in sorted(package_dir.rglob("*.py"))
        for module in _imported_modules(path)
        if _is_forbidden(module)
    ]
    assert not violations, f"midrun.debugger must stay out-of-process: {violations}"
