export ReLU

"""
$(TYPEDEF)

Represents a ReLU operation.

`p(x)` is shorthand for [`relu(x)`](@ref) when `p` is an instance of
`ReLU`.
"""
struct ReLU <: Layer
    tightening_algorithms::AbstractArray{<:MIPVerify.TighteningAlgorithm}
end

ReLU() = ReLU(DEFAULT_TIGHTENING_ALGORITHM_SEQUENCE)

function ReLU(ta::MIPVerify.TighteningAlgorithm)
    ReLU([ta])
end

Base.hash(a::ReLU, h::UInt) = hash(:ReLU, h)

function Base.show(io::IO, p::ReLU)
    print(io, "ReLU()")
end

(p::ReLU)(x::Array{<:Real}) = relu(x)
(p::ReLU)(x::Array{<:JuMPLinearType}) = (info(MIPVerify.LOGGER, "Applying $p ..."); relu(x, tas = p.tightening_algorithms))
