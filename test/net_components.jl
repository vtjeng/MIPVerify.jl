using Base.Test

@testset "net_components/" begin
    include("net_components/core_ops.jl")
    include("net_components/layers.jl")
    include("net_components/nets.jl")
end