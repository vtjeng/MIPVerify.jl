export ConvolutionLayerParameters

"""
$(TYPEDEF)

Stores parameters for a convolution layer consisting of a convolution, followed by
max-pooling, and a ReLU activation function.

`p(x)` is shorthand for [`convolution_layer(x, p)`](@ref) when `p` is an instance of
`ConvolutionLayerParameters`.

## Fields:
$(FIELDS)
"""
@auto_hash_equals struct ConvolutionLayerParameters{T<:Real, U<:Real} <: StackableLayerParameters
    conv2dparams::Conv2DParameters{T, U}
    maxpoolparams::PoolParameters{4}

    function ConvolutionLayerParameters{T, U}(conv2dparams::Conv2DParameters{T, U}, maxpoolparams::PoolParameters{4}) where {T<:Real, U<:Real}
        @assert maxpoolparams.pooling_function == MIPVerify.maximum
        return new(conv2dparams, maxpoolparams)
    end

end

"""
$(SIGNATURES)

Convenience function to create a [`ConvolutionLayerParameters`](@ref) struct with the
specified filter, bias, and strides for the max-pooling operation. 
"""
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

"""
$(SIGNATURES)

Computes the result of convolving `x` with `params.conv2dparams`, pooling the resulting
output with the pooling function and strides specified in `params.maxpoolparams`, and
passing the output through a ReLU activation function.
"""
function convolution_layer(
    x::Array{<:JuMPReal, 4},
    params::ConvolutionLayerParameters)
    x_relu = relu.(x |> params.conv2dparams |> params.maxpoolparams)
    return x_relu
end

(p::ConvolutionLayerParameters)(x::Array{<:JuMPReal, 4}) = convolution_layer(x, p)
