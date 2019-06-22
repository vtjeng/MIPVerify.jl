export Reshape

"""
$(TYPEDEF)

Represents a ReLU operation.

`p(x)` is shorthand for [`relu(x)`](@ref) when `p` is an instance of
`ReLU`.
"""
struct Reshape <: Layer
    shape::Array{Integer, 1}
end

function Base.show(io::IO, p::Reshape)
    print(io, "Reshape(shape: $(p.shape)")
end

function apply(p::Reshape, x::Array{<:JuMPReal})
    m = reshape(x, p.shape)
    return m
end

(p::Reshape)(x::Array{<:JuMPReal}) = apply(p, x)