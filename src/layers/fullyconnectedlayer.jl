function fullyconnectedlayer(
    x::Array{T, 1}, 
    params::FullyConnectedLayerParameters) where {T<:JuMPReal}
    return relu.(x |> params.mmparams)
end

(p::FullyConnectedLayerParameters)(x::Array{T, 1}) where {T<:JuMPReal} = fullyconnectedlayer(x, p)