export DWT_Pooling

"""
$(TYPEDEF)

Represents a DWT_Pooling operation.
"""
struct DWT_Pooling <: Layer
    perm::AbstractArray
end

function Base.show(io::IO, p::DWT_Pooling)
    print(io, "DWT_Pooling(perm: $(p.perm)")
end

function apply(x::Array{<:JuMPReal}, p::DWT_Pooling)
    m = permutedims(x, p.perm)
    return m
end

(p::DWT_Pooling)(x::Array{<:JuMPReal}) = apply(x, p)