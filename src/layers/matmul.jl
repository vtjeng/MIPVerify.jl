function matmul(
    x::Array{<:JuMPReal, 1}, 
    params::MatrixMultiplicationParameters)
    return params.matrix.'*x .+ params.bias
end

(p::MatrixMultiplicationParameters)(x::Array{<:JuMPReal, 1}) = matmul(x, p)