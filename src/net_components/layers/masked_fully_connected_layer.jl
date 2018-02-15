"""
Same as a fully connected layer, but with an additional mask that controls whether a ReLU
is applied to each output. 

  1) If the value of the mask is <0 (i.e. input is assumed to be always non-positive), the 
     output is set at 0.
  2) If the value of the mask is 0 (i.e. input can take both positive and negative values),
     the output is rectified.
  3) If the value of the mask is >0 (i.e. input is assumed to be always non-negative), the 
     output is set as the value of the input, without any rectification.
"""
@auto_hash_equals struct MaskedFullyConnectedLayerParameters{T<:Real, U<:Real, V<:Real} <: StackableLayerParameters
    mmparams::MatrixMultiplicationParameters{T, U}
    mask::Array{V, 1}

    function MaskedFullyConnectedLayerParameters{T, U, V}(
        mmparams::MatrixMultiplicationParameters{T, U},
        mask::Array{V, 1}) where {T<:Real, U<:Real, V<:Real}
        bias_height = size(mmparams.bias) 
        mask_height = size(mask)
        @assert(
            bias_height == mask_height,
            "Size of output layer, $bias_height, does not match size of provided mask, $mask_height."
        )
        return new(mmparams, mask)
    end
end

function MaskedFullyConnectedLayerParameters{T<:Real, U<:Real, V<:Real}(matrix::Array{T, 2}, bias::Array{U, 1}, mask::Array{V, 1})
    MaskedFullyConnectedLayerParameters{T, U, V}(
        MatrixMultiplicationParameters(matrix, bias),
        mask
    )
end

function Base.show(io::IO, p::MaskedFullyConnectedLayerParameters)
    num_zeroed_units = count(p.mask .< 0)
    num_passthrough_units = count(p.mask .> 0)
    num_rectified_units = output_size(p.mmparams) - num_zeroed_units - num_passthrough_units
    print(io,
        "masked fully connected layer with $(p.mmparams |> input_size) inputs and $(p.mmparams |> output_size) output units ($(num_zeroed_units) zeroed, $(num_passthrough_units) as-is, $(num_rectified_units) rectified)."
    )
end

function masked_fully_connected_layer(
    x::Array{<:JuMPReal, 1}, 
    params::MaskedFullyConnectedLayerParameters)
    return masked_relu.(x |> params.mmparams, params.mask)
end

(p::MaskedFullyConnectedLayerParameters)(x::Array{<:JuMPReal, 1}) = masked_fully_connected_layer(x, p)