export DWT_Pooling

"""
$(TYPEDEF)

Represents a DWT_Pooling operation.
"""
struct DWT_Pooling <: Layer

end

function Base.show(io::IO, p::DWT_Pooling)
    print(io, "DWT_Pooling")
end

function apply(x::Array{<:JuMPReal})
    xt = dwt(x, wavelet(WT.haar))
    return m
end

(p::DWT_Pooling)(x::Array{<:JuMPReal}) = apply(x)