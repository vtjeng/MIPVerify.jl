#!/bin/bash

# Determining root directory of the repo: https://stackoverflow.com/a/957978
repo_root=$(git rev-parse --show-toplevel)

(
  cd "${repo_root}" &&
  julia -e '
    using Pkg;
    Pkg.activate(tempname());
    Pkg.add(PackageSpec(name="JuliaFormatter", version="1.0.2"));
    using JuliaFormatter;
    format(".", verbose=true)
  '
)

# Format benchmark analysis Python with the same pinned Ruff version that CI uses
# (see format-check-python in .github/workflows/CI.yml).
if command -v uvx > /dev/null; then
  (cd "${repo_root}" && uvx ruff@0.15.21 format benchmarks/analysis/)
else
  echo "WARNING: uvx not found; skipping Ruff formatting of Python files." >&2
fi

# Format Markdown and YAML with the same pinned Prettier version that CI uses
# (see format-check-prettier in .github/workflows/CI.yml).
if command -v npx > /dev/null; then
  (cd "${repo_root}" && npx --yes prettier@3.9.5 --write "**/*.{md,yaml}")
else
  echo "WARNING: npx not found; skipping Prettier formatting of Markdown and YAML files." >&2
fi
