using Base.Test

@testset "layers/" begin
    include("conv2d.jl")
    include("core_ops.jl")
    include("net_parameters.jl")
    include("pool.jl")
end