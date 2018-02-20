using JuMP
using AutoHashEquals

export LayerParameters, StackableLayerParameters, NeuralNetParameters

"""
$(TYPEDEF)
"""
JuMPReal = Union{Real, JuMP.AbstractJuMPScalar}

include("core_ops.jl")

"""
$(TYPEDEF)

Supertype for all types storing the parameters of each layer. Inherit from this
to specify your own custom type of layer. Each implementation is expected to:
    
1. Implement a callable specifying the output when any input of type [`JuMPReal`](@ref) is provided.
"""
abstract type LayerParameters end

"""
$(TYPEDEF)

Supertype for all [`LayerParameters`](@ref) that can be logically applied in
sequence.

An array of `StackableLayerParameters` is interpreted as that array of layer being applied
to the input sequentially, starting from the leftmost layer. (In functional programming
terms, this can be thought of as a sort of `fold`).
"""
abstract type StackableLayerParameters <: LayerParameters end

(ps::Array{<:Union{StackableLayerParameters}, 1})(x::Array{<:JuMPReal}) = (
    length(ps) == 0 ? x : ps[2:end](ps[1](x))
)

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

"""
$(TYPEDEF)

Supertype for all types storing the parameters of a neural net. Inherit from this
to specify your own custom architecture of [`LayerParameters`](@ref). Each implementation
is expected to:

1. Implement a callable specifying the output when any input of type [`JuMPReal`](@ref) is provided
2. Have a `UUID` field for the name of the neural network.
"""
abstract type NeuralNetParameters end

include("nets/standard_neural_net.jl")
include("nets/masked_fc_net.jl")