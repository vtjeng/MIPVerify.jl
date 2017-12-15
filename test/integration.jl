using Base.Test

@testset "integration tests" begin
    include("integration/generated_weights.jl")
    if Pkg.installed("Gurobi") != nothing
        # Skip these tests if Gurobi is not installed.
        # The corresponding networks are too large for CBC to deal with.
        include("integration/trained_weights.jl")
    end
end