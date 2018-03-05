export Flatten

@auto_hash_equals struct Flatten{T<:Int} <: Layer
    n_dim::Int
    perm::AbstractArray{T}

    function Flatten{T}(n_dim::Int, perm::AbstractArray{T}) where {T<:Int}
        @assert(
            length(perm) == n_dim,
            "Number of dimensions to be permuted, $(length(perm)) does not match expected value, $(n_dim)."
        )
        return new(n_dim, perm)
    end
end

function Flatten(n_dim::Int, perm::AbstractArray{T}) where {T<:Int}
    Flatten{T}(n_dim, perm)
end

function Flatten(n_dim::Int)::Flatten
    Flatten(n_dim, n_dim:-1:1)
end

function Flatten(perm::AbstractArray{T})::Flatten where {T<:Int}
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
Permute dimensions of array because Python flattens arrays in the opposite order.
"""
function flatten(x::Array{T, N}, perm::AbstractArray{U}) where {T, N, U<:Int}
    @assert length(perm) == N
    return permutedims(x, perm)[:]
end

(p::Flatten)(x::Array{<:JuMPReal}) = flatten(x, p.perm)