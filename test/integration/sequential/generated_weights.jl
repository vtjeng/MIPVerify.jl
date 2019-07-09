using Test

@testset "generated_weights" begin
    include("generated_weights/fc+softmax.jl")
    include("generated_weights/mfc+mfc+softmax.jl")
    include("generated_weights/conv+fc+softmax.jl")

    if Base.find_package("Gurobi") != nothing
        # Skip these tests if Gurobi is not installed.
        include("generated_weights/conv+conv+fc+softmax.jl")
    end
end