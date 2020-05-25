#!/bin/bash

# per https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

(cd $DIR/.. && julia -e 'using Pkg; Pkg.activate(tempname()); Pkg.add(PackageSpec(name="JuliaFormatter", version="0.5.4")); using JuliaFormatter; format(".", verbose=true)')
