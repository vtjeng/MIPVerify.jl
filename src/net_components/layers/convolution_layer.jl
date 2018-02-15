@auto_hash_equals struct ConvolutionLayerParameters{T<:Real, U<:Real} <: StackableLayerParameters
    conv2dparams::Conv2DParameters{T, U}
    maxpoolparams::PoolParameters{4}

    function ConvolutionLayerParameters{T, U}(conv2dparams::Conv2DParameters{T, U}, maxpoolparams::PoolParameters{4}) where {T<:Real, U<:Real}
        @assert maxpoolparams.pooling_function == MIPVerify.maximum
        return new(conv2dparams, maxpoolparams)
    end

end

function ConvolutionLayerParameters{T<:Real, U<:Real}(filter::Array{T, 4}, bias::Array{U, 1}, strides::NTuple{4, Int})
    ConvolutionLayerParameters{T, U}(Conv2DParameters(filter, bias), MaxPoolParameters(strides))
end

function check_size(params::ConvolutionLayerParameters, sizes::NTuple{4, Int})::Void
    check_size(params.conv2dparams, sizes)
end

function Base.show(io::IO, p::ConvolutionLayerParameters)
    print(io,
        "convolution layer. $(p.conv2dparams), followed by $(p.maxpoolparams), and a ReLU activation function.",
    )
end

function convolution_layer(
    x::Array{<:JuMPReal, 4},
    params::ConvolutionLayerParameters)
    x_relu = relu.(x |> params.conv2dparams |> params.maxpoolparams)
    return x_relu
end

(p::ConvolutionLayerParameters)(x::Array{<:JuMPReal, 4}) = convolution_layer(x, p)
