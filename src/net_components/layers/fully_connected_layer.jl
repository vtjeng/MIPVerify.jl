export FullyConnectedLayerParameters

"""
$(TYPEDEF)

Stores parameters for a fully connected layer consisting of a matrix multiplication 
followed by a ReLU activation function.

`p(x)` is shorthand for [`fully_connected_layer(x, p)`](@ref) when `p` is an instance of
`FullyConnectedLayerParameters`.

## Fields:
$(FIELDS)
"""
@auto_hash_equals struct FullyConnectedLayerParameters{T<:Real, U<:Real} <: StackableLayerParameters
    mmparams::MatrixMultiplicationParameters{T, U}
end

function FullyConnectedLayerParameters(matrix::Array{T, 2}, bias::Array{U, 1}) where {T<:Real, U<:Real}
    FullyConnectedLayerParameters(MatrixMultiplicationParameters(matrix, bias))
end

function Base.show(io::IO, p::FullyConnectedLayerParameters)
    print(io,
        "fully connected layer with $(p.mmparams |> input_size) inputs and $(p.mmparams |> output_size) output units, and a ReLU activation function."
    )
end

"""
Computes the result of multiplying x by `params.mmparams`, and 
passing the output through a ReLU activation function.
"""
function fully_connected_layer(
    x::Array{<:JuMPReal, 1}, 
    params::FullyConnectedLayerParameters)
    return relu.(x |> params.mmparams)
end

(p::FullyConnectedLayerParameters)(x::Array{<:JuMPReal, 1}) = fully_connected_layer(x, p)