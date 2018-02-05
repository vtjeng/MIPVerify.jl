function convlayer(
    x::Array{<:JuMPReal, 4},
    params::ConvolutionLayerParameters)
    x_relu = relu.(x |> params.conv2dparams |> params.maxpoolparams)
    return x_relu
end

(p::ConvolutionLayerParameters)(x::Array{<:JuMPReal, 4}) = convlayer(x, p)
