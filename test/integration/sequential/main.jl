using Base.Test

@testset "sequential" begin
    include("generated_weights/main.jl")
    if Pkg.installed("Gurobi") != nothing
        # Skip these tests if Gurobi is not installed.
        # The corresponding networks are too large for CBC to deal with.
        include("trained_weights/main.jl")
    end
end