export MaskedReLU

"""
Applies a ReLU activation, with the `mask` controlling how the ReLU is applied to each output.

  1) If the value of the mask is <0 (i.e. input is assumed to be always non-positive), the 
     output is set at 0.
  2) If the value of the mask is 0 (i.e. input can take both positive and negative values),
     the output is rectified.
  3) If the value of the mask is >0 (i.e. input is assumed to be always non-negative), the 
     output is set as the value of the input, without any rectification.
"""
@auto_hash_equals struct MaskedReLU{T<:Real} <: Layer
    mask::Array{T}
end

function Base.show(io::IO, p::MaskedReLU)
    num_zeroed_units = count(p.mask .< 0)
    num_passthrough_units = count(p.mask .> 0)
    num_rectified_units = length(p.mask) - num_zeroed_units - num_passthrough_units
    print(io,
        "MaskedReLU with expected input size $(size(p.mask)). ($(num_zeroed_units) zeroed, $(num_passthrough_units) as-is, $(num_rectified_units) rectified)."
    )
end

(p::MaskedReLU)(x::Array{<:JuMPReal}) = masked_relu.(x, p.mask)