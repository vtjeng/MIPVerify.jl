using Base.Test

@testset "generated_weights" begin
    include("generated_weights/conv+softmax.jl")
    include("generated_weights/conv+fc+softmax.jl")
    include("generated_weights/conv+conv+fc+softmax.jl")
end