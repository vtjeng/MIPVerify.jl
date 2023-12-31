using Test
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "layers/" begin
    include("layers/conv2d.jl")
    include("layers/flatten.jl")
    include("layers/linear.jl")
    include("layers/masked_relu.jl")
    include("layers/pool.jl")
    include("layers/relu.jl")
    include("layers/zero.jl")
end
