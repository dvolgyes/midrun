# Building release archives

Run the manual **Build release archives** workflow on the intended release commit.

The workflow builds from `git archive` without publishing either archive.

It rejects a stale `uv.lock` and checks both archives for missing or extra package files.

**Do not distribute the pre-existing local 0.1.1 archives:** the local wheel contains training run logs.

To build and retain a verified pair locally, run the following from the repository root with uv 0.13.9:

```bash
set -euo pipefail
src=$(mktemp -d)
out=$(mktemp -d)
git archive HEAD | tar -x -C "$src"
(cd "$src" && uv lock --check && uv build --out-dir "$out" && uv run --script tools/check_release.py "$out")
printf 'Verified release archives: %s\n' "$out"
```

Use only the new output directory after the checks succeed.
