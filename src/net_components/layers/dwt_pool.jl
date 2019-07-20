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
    println("apply DWT_Pooling to shape", size(x))
    out = zeros((1, Int(size(x)[2]/2), Int(size(x)[3]/2), 3))
    for i=1:size(x)[3]
        wt = wavelet(WT.haar)
        wt = WT.scale(wt, 1/sqrt(2))
        x1 = dwt(x[1,:,:,i], wt, 1)
        println(size(x1))
        out[1,:,:,i] = x1[1:Int(size(x1)[2]/2), 1:Int(size(x1)[3]/2)]
    end
    return out
end

(p::DWT_Pooling)(x::Array{<:JuMPReal}) = apply(x)