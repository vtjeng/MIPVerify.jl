export SoftmaxParameters

"""
$(TYPEDEF)

Stores parameters for a softmax layer consisting of a matrix multiplication with _no_
activation function.

This simply wraps [`MatrixMultiplicationParameters`](@ref) to ensure that it is
distinguishable as a softmax layer.

## Fields:
$(FIELDS)
"""
@auto_hash_equals struct SoftmaxParameters{T<:Real, U<:Real} <: LayerParameters
    mmparams::MatrixMultiplicationParameters{T, U}
end

function SoftmaxParameters(matrix::Array{T, 2}, bias::Array{U, 1}) where {T<:Real, U<:Real}
    SoftmaxParameters(MatrixMultiplicationParameters(matrix, bias))
end

function Base.show(io::IO, p::SoftmaxParameters)
    print(io,
        "softmax layer with $(p.mmparams |> input_size) inputs and $(p.mmparams |> output_size) output units."
    )
end

(p::SoftmaxParameters)(x::Array{<:JuMPReal, 1}) = p.mmparams(x)