using Base.Test

@testset "nets/" begin
    include("standard_neural_net.jl")
    include("masked_fc_net.jl")
end