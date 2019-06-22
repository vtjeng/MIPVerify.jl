export Permutedims

"""
$(TYPEDEF)

Represents a permutedims operation.
"""
struct Permutedims <: Layer
    perm::AbstractArray
end

function Base.show(io::IO, p::Permutedims)
    print(io, "Permutedims(perm: $(p.perm)")
end

function apply(x::Array{<:JuMPReal}, p::Permutedims)
    m = permutedims(x, p.perm)
    return m
end

(p::Permutedims)(x::Array{<:JuMPReal}) = apply(x, p)