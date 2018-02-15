using Base.Test

@testset "integration tests" begin
    include("standard_neural_net/main.jl")
    include("masked_fc_net/main.jl")
end