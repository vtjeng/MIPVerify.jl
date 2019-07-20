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
    wt = wavelet(WT.haar)
    wt = WT.scale(wt, 1/sqrt(2))
    x1 = dwt(x, wt, 1)
    out = x1[1:Int(size(x1)[1]/2), 1:Int(size(x1)[2]/2)]
    return out
end

(p::DWT_Pooling)(x::Array{<:JuMPReal}) = apply(x)