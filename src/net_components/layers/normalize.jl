export Normalize

"""
$(TYPEDEF)

Represents a Normalization operation.
"""
@auto_hash_equals struct Normalize <: Layer
    mean::Array{Real, 1}
    std::Array{Real, 1}
end

function Base.show(io::IO, p::Normalize)
    print(io, "Normalize(means: $(p.mean), stds: $(p.std))")
end

function apply(p::Normalize, x::Array{<:JuMPReal})
    padded_shape = (ones(Int, ndims(x)-1)..., length(p.mean))
    m = reshape(p.mean, padded_shape)
    s = reshape(p.std, padded_shape)
    output = (x.-m)./s
    return output
end

(p::Normalize)(x::Array{<:JuMPReal}) = apply(p, x)
