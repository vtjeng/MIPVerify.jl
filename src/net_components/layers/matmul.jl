@auto_hash_equals struct MatrixMultiplicationParameters{T<:Real, U<:Real} <: LayerParameters
    matrix::Array{T, 2}
    bias::Array{U, 1}

    function MatrixMultiplicationParameters{T, U}(matrix::Array{T, 2}, bias::Array{U, 1}) where {T<:Real, U<:Real}
        (matrix_width, matrix_height) = size(matrix)
        bias_height = length(bias)
        @assert(
            matrix_height == bias_height,
            "Number of output channels in matrix, $matrix_height, does not match number of output channels in bias, $bias_height."
        )
        return new(matrix, bias)
    end

end

function MatrixMultiplicationParameters(matrix::Array{T, 2}, bias::Array{U, 1}) where {T<:Real, U<:Real}
    MatrixMultiplicationParameters{T, U}(matrix, bias)
end

function check_size(params::MatrixMultiplicationParameters, sizes::NTuple{2, Int})::Void
    check_size(params.matrix, sizes)
    check_size(params.bias, (sizes[end], ))
end

input_size(p::MatrixMultiplicationParameters) = size(p.matrix)[1]
output_size(p::MatrixMultiplicationParameters) = size(p.matrix)[2]

function matmul(
    x::Array{<:JuMPReal, 1}, 
    params::MatrixMultiplicationParameters)
    return params.matrix.'*x .+ params.bias
end

(p::MatrixMultiplicationParameters)(x::Array{<:JuMPReal, 1}) = matmul(x, p)