export Flatten

"""
$(TYPEDEF)

Represents a flattening operation.

`p(x)` is shorthand for [`flatten(x, p.perm)`](@ref) when `p` is an instance of
`Flatten`.

## Fields:
$(FIELDS)
"""
@auto_hash_equals struct Flatten{T<:Integer} <: Layer
    n_dim::Integer
    perm::AbstractArray{T}

    function Flatten{T}(n_dim::Integer, perm::AbstractArray{T}) where {T<:Integer}
        @assert(
            length(perm) == n_dim,
            "Number of dimensions to be permuted, $(length(perm)) does not match expected value, $(n_dim)."
        )
        return new(n_dim, perm)
    end
end

function Flatten(n_dim::Integer, perm::AbstractArray{T}) where {T<:Integer}
    Flatten{T}(n_dim, perm)
end

function Flatten(n_dim::Integer)::Flatten
    Flatten(n_dim, n_dim:-1:1)
end

function Flatten(perm::AbstractArray{T})::Flatten where {T<:Integer}
    if !all(sort(perm) .== 1:length(perm))
        throw(DomainError("$perm is not a valid permutation."))
    end
    Flatten(length(perm), perm)
end

function Base.show(io::IO, p::Flatten)
    print(io,
        "Flatten(): flattens $(p.n_dim) dimensional input, with dimensions permuted according to the order $(p.perm |> collect)"
    )
end

"""
Permute dimensions of array in specified order, then flattens the array.
"""
function flatten(x::Array{T, N}, perm::AbstractArray{U}) where {T, N, U<:Integer}
    @assert all(sort(perm) .== 1:N)
    return permutedims(x, perm)[:]
end

(p::Flatten)(x::Array{<:Real}) = flatten(x, p.perm)
(p::Flatten)(x::Array{<:JuMPLinearType}) = (info(MIPVerify.LOGGER, "Applying Flatten() ... "); flatten(x, p.perm))