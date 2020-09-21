export Zero

"""
$(TYPEDEF)

Always outputs exactly zero.
"""
struct Zero <: Layer end

function Base.show(io::IO, p::Zero)
    print(io, "Zero()")
end

(p::Zero)(x::Array{<:JuMPReal}) = 0
