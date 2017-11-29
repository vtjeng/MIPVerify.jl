function convlayer(
    x::Array{T, 4},
    params::ConvolutionLayerParameters) where {T<:JuMPReal}
    x_relu = relu.(x |> params.conv2dparams |> params.maxpoolparams)
    return x_relu
end

(p::ConvolutionLayerParameters)(x::Array{T, 4}) where {T<:JuMPReal} = convlayer(x, p)
