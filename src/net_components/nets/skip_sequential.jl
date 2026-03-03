export SkipSequential

"""
$(TYPEDEF)

Represents a sequential (feed-forward) neural net, with `layers` ordered from input
to output. Unlike a regular `Sequential` network, this network type supports [`SkipBlock`](@ref)s,
which can take input from multiple previous layers. When a `SkipBlock` is encountered, it receives
an array of outputs from preceding layers, allowing for skip connections and residual architectures.

## Fields:
$(FIELDS)
"""
struct SkipSequential <: NeuralNet
    layers::Array{Layer,1}
    UUID::String
end

function Base.show(io::IO, p::SkipSequential)
    println(io, "skip sequential net $(p.UUID)")
    for (index, value) in enumerate(p.layers)
        println(io, "  ($index) $value")
    end
end

function apply(p::SkipSequential, x::AbstractArray{<:JuMPReal})
    xs = Vector{AbstractArray{<:JuMPReal}}(undef, 1)
    xs[1] = x
    for layer in p.layers
        if layer isa SkipBlock
            output = layer(xs)
        else
            output = layer(last(xs))
        end
        # We keep track of the intermediate outputs for skip connections
        push!(xs, output)
    end
    return last(xs)
end

(p::SkipSequential)(x::AbstractArray{<:JuMPReal}) = apply(p, x)
