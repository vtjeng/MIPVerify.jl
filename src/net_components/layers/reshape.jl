export Reshape

"""
$(TYPEDEF)

Represents a ReLU operation.

`p(x)` is shorthand for [`relu(x)`](@ref) when `p` is an instance of
`ReLU`.
"""
struct Reshape <: Layer
    dims::Dims
end

function Base.show(io::IO, p::Reshape)
    print(io, "Reshape(shape: $(p.shape)")
end

function apply(x::Array{<:JuMPReal}, p::Reshape)
    m = reshape(x, p.dims)
    return m
end

(p::Reshape)(x::Array{<:JuMPReal}) = apply(x, p)