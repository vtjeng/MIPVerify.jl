export Reshape

"""
$(TYPEDEF)

Represents a Reshape operation.
"""
struct Reshape <: Layer
    dims::Dims
end

function Base.show(io::IO, p::Reshape)
    print(io, "Reshape(shape: $(p.dims)")
end

function apply(x::Array{<:JuMPReal}, p::Reshape)
    m = reshape(x, p.dims)
    return m
end

(p::Reshape)(x::Array{<:JuMPReal}) = apply(x, p)