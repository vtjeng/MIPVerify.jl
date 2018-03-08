using Base.Test

@testset "sequential" begin
    include("sequential/generated_weights.jl")
    if Pkg.installed("Gurobi") != nothing
        # Skip these tests if Gurobi is not installed.
        # The corresponding networks are too large for CBC to deal with.
        include("sequential/trained_weights.jl")
    end
end