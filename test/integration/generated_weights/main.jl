using Base.Test

@testset "generated_weights" begin
    include("conv+fc+softmax.jl")

    if Pkg.installed("Gurobi") != nothing
        # Skip these tests if Gurobi is not installed.
        include("conv+softmax.jl")
        include("conv+conv+fc+softmax.jl")
    end
end