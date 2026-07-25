#!/bin/bash

# Determining root directory of the repo: https://stackoverflow.com/a/957978
repo_root=$(git rev-parse --show-toplevel)

julia_only=false
if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && $1 == "--julia-only" ]]; then
    julia_only=true
  else
    echo "Usage: $0 [--julia-only]" >&2
    exit 2
  fi
fi

(
  cd "${repo_root}" &&
  julia -e '
    using Pkg;
    Pkg.activate(tempname());
    # renovate: datasource=github-tags depName=JuliaEditorSupport/JuliaFormatter.jl
    Pkg.add(PackageSpec(name="JuliaFormatter", version="1.0.62"));
    using JuliaFormatter;
    format(".", verbose=true)
  '
)

julia_formatter_status=$?
if [[ "${julia_only}" == true ]]; then
  exit "${julia_formatter_status}"
fi

# Format benchmark analysis Python with the same pinned Ruff version that CI uses
# (see format-check-python in .github/workflows/CI.yml).
if command -v uvx > /dev/null; then
  # renovate: datasource=pypi depName=ruff
  (cd "${repo_root}" && uvx ruff@0.15.21 format benchmarks/analysis/)
else
  echo "WARNING: uvx not found; skipping Ruff formatting of Python files." >&2
fi

# Format Markdown and YAML with the same pinned Prettier version that CI uses
# (see format-check-prettier in .github/workflows/CI.yml).
if command -v npx > /dev/null; then
  # renovate: datasource=npm depName=prettier
  (cd "${repo_root}" && npx --yes prettier@3.9.6 --write "**/*.{md,yaml}")
else
  echo "WARNING: npx not found; skipping Prettier formatting of Markdown and YAML files." >&2
fi
