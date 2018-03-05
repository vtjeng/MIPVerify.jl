using Base.Test

@testset "layers/" begin
    include("conv2d.jl")
    include("flatten.jl")
    include("linear.jl")
    include("masked_relu.jl")
    include("pool.jl")
    include("relu.jl")
end