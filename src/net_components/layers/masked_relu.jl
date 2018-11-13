export MaskedReLU

"""
$(TYPEDEF)

Represents a masked ReLU activation, with `mask` controlling how the ReLU is applied to
each output.

`p(x)` is shorthand for [`masked_relu(x, p.mask)`](@ref) when `p` is an instance of
`MaskedReLU`.

## Fields:
$(FIELDS)
"""
@auto_hash_equals struct MaskedReLU{T<:Real} <: Layer
    mask::Array{T}
    tightening_algorithms::AbstractArray{<:MIPVerify.TighteningAlgorithm}
end

function MaskedReLU(mask::Array{T}) where {T<:Real}
    MaskedReLU{T}(mask, MIPVerify.DEFAULT_TIGHTENING_ALGORITHM_SEQUENCE)
end

function MaskedReLU(mask::Array{T}, ta::MIPVerify.TighteningAlgorithm) where {T<:Real}
    MaskedReLU{T}(mask, [ta])
end

# TODO (vtjeng): Remove
# function MaskedReLU(mask::Array{T}, ta::AbstractArray{<:MIPVerify.TighteningAlgorithm}) where {T<:Real}
#     MaskedReLU{T}(mask, ta)
# end

function Base.show(io::IO, p::MaskedReLU)
    num_zeroed_units = count(p.mask .< 0)
    num_passthrough_units = count(p.mask .> 0)
    num_rectified_units = length(p.mask) - num_zeroed_units - num_passthrough_units
    print(io,
        "MaskedReLU with expected input size $(size(p.mask)). ($(num_zeroed_units) zeroed, $(num_passthrough_units) as-is, $(num_rectified_units) rectified)."
    )
end

(p::MaskedReLU)(x::Array{<:Real}) = masked_relu(x, p.mask)
(p::MaskedReLU)(x::Array{<:JuMPLinearType}) = (info(MIPVerify.LOGGER, "Applying $p ... "); masked_relu(x, p.mask; tas = p.tightening_algorithms))