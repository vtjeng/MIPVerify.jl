using Base.Test

@testset "layers/" begin
    include("layers/conv2d.jl")
    include("layers/flatten.jl")
    include("layers/linear.jl")
    include("layers/masked_relu.jl")
    include("layers/pool.jl")
    include("layers/relu.jl")
end