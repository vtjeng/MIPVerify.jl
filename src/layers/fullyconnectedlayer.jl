function fullyconnectedlayer(
    x::Array{<:JuMPReal, 1}, 
    params::FullyConnectedLayerParameters)
    return relu.(x |> params.mmparams)
end

(p::FullyConnectedLayerParameters)(x::Array{<:JuMPReal, 1}) = fullyconnectedlayer(x, p)