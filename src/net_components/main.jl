using JuMP
using AutoHashEquals

JuMPReal = Union{Real, JuMP.AbstractJuMPScalar}

include("core_ops.jl")

abstract type LayerParameters end
abstract type StackableLayerParameters <: LayerParameters end

function check_size(input::AbstractArray, expected_size::NTuple{N, Int})::Void where {N}
    input_size = size(input)
    @assert input_size == expected_size "Input size $input_size did not match expected size $expected_size."
end

include("layers/conv2d.jl")
include("layers/pool.jl")
include("layers/matmul.jl")
include("layers/softmax.jl")
include("layers/convolution_layer.jl")
include("layers/fully_connected_layer.jl")
include("layers/masked_fully_connected_layer.jl")

(ps::Array{<:Union{StackableLayerParameters}, 1})(x::Array{<:JuMPReal}) = (
    length(ps) == 0 ? x : ps[2:end](ps[1](x))
)

abstract type NeuralNetParameters end

include("nets/standard_neural_net.jl")
include("nets/masked_fc_net.jl")