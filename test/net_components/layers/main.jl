using Base.Test

@testset "layers/" begin
    include("conv2d.jl")
    include("convolution_layer.jl")
    include("fully_connected_layer.jl")
    include("matmul.jl")
    include("pool.jl")
    include("softmax.jl")
end