# Smoke-tests the example notebooks by executing them via NBInclude.
# This catches API breakage and runtime errors but does not verify output values —
# correctness assertions live in test/runtests.jl.
#
# Set NOTEBOOK env var to run a single notebook (used by CI matrix strategy).

using Test
using NBInclude

const EXAMPLES_DIR = joinpath(@__DIR__, "..")

notebooks_to_run = if haskey(ENV, "NOTEBOOK")
    [ENV["NOTEBOOK"]]
else
    filter(f -> endswith(f, ".ipynb"), readdir(EXAMPLES_DIR))
end

@assert !isempty(notebooks_to_run) "No notebooks found in $EXAMPLES_DIR"

@testset "Example Notebooks" begin
    for nb in notebooks_to_run
        @testset "$nb" begin
            let nb_path = joinpath(EXAMPLES_DIR, nb)
                @nbinclude(nb_path; softscope = true)
            end
        end
    end
end
