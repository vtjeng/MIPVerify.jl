export Linear

"""
$(TYPEDEF)

Represents matrix multiplication.

`p(x)` is shorthand for [`matmul(x, p)`](@ref) when `p` is an instance of
`Linear`.

## Fields:
$(FIELDS)
"""
@auto_hash_equals struct Linear{T<:Real, U<:Real} <: Layer
    matrix::Array{T, 2}
    bias::Array{U, 1}

    function Linear{T, U}(matrix::Array{T, 2}, bias::Array{U, 1}) where {T<:Real, U<:Real}
        (matrix_width, matrix_height) = size(matrix)
        bias_height = length(bias)
        @assert(
            matrix_height == bias_height,
            "Number of output channels in matrix, $matrix_height, does not match number of output channels in bias, $bias_height."
        )
        return new(matrix, bias)
    end

end

function Linear(matrix::Array{T, 2}, bias::Array{U, 1}) where {T<:Real, U<:Real}
    Linear{T, U}(matrix, bias)
end

function Base.show(io::IO, p::Linear)
    input_size = size(p.matrix)[1]
    output_size = size(p.matrix)[2]
    print(io,
        "Linear($input_size -> $output_size)"
    )
end

function check_size(params::Linear, sizes::NTuple{2, Int})::Void
    check_size(params.matrix, sizes)
    check_size(params.bias, (sizes[end], ))
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`.
"""
function matmul(
    x::Array{<:JuMPReal, 1}, 
    params::Linear)
    return params.matrix.'*x .+ params.bias
end

(p::Linear)(x::Array{<:JuMPReal}) = "Linear() layers work only on one-dimensional input. You likely forgot to add a Flatten() layer before your first linear layer." |> ArgumentError |> throw

(p::Linear)(x::Array{<:Real, 1}) = matmul(x, p)
(p::Linear)(x::Array{<:JuMP.AbstractJuMPScalar, 1}) = (info(MIPVerify.LOGGER, "Applying $p ... "); matmul(x, p))