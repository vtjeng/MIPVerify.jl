#!/bin/bash

# Determining root directory of the repo: https://stackoverflow.com/a/957978
repo_root=$(git rev-parse --show-toplevel)

(
  cd "${repo_root}" &&
  julia -e '
    using Pkg;
    Pkg.activate(tempname());
    Pkg.add(PackageSpec(name="JuliaFormatter", version="0.5.4")); 
    using JuliaFormatter;
    format(".", verbose=true)
  '
)
