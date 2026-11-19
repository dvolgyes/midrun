#!/usr/bin/env bash
# Runs the midrun playground in the foreground: Lightning/PyTorch logs print to
# this terminal, and Ctrl+C stops training gracefully. Inspect the live run from a
# second terminal using the REST/FUSE commands printed at startup.
set -euo pipefail

CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec uv run --group examples python -m examples.playground.run "$@"
