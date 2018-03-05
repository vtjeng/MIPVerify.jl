export ReLU

struct ReLU <: Layer end

Base.hash(a::ReLU, h::UInt) = hash(:ReLU, h)

function Base.show(io::IO, p::ReLU)
    print(io, "ReLU()")
end

(p::ReLU)(x::Array{<:JuMPReal}) = relu.(x)