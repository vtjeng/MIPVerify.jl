function matmul(
    x::Array{T, 1}, 
    params::MatrixMultiplicationParameters) where {T<:JuMPReal}
    return params.matrix.'*x .+ params.bias
end

(p::MatrixMultiplicationParameters)(x::Array{T, 1}) where {T<:JuMPReal} = matmul(x, p)