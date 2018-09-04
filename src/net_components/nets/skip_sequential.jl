export SkipSequential

"""
$(TYPEDEF)

Represents a sequential (feed-forward) neural net, with `layers` ordered from input
to output.

## Fields:
$(FIELDS)
"""
@auto_hash_equals struct SkipSequential <: NeuralNet
    layers::Array{Layer, 1}
    UUID::String
end

function Base.show(io::IO, p::SkipSequential)
    println(io, "skip sequential net $(p.UUID)")
    for (index, value) in enumerate(p.layers)
        println(io, "  ($index) $value")
    end
end

# TODO (vtjeng): Think about the types carefully.
function apply(p::SkipSequential, x::Array{<:JuMPReal})
    xs::Array{Array} = [x]
    for layer in p.layers
        if typeof(layer)<:SkipBlock
            output = layer(xs)
        else
            output = layer(last(xs))
        end
        push!(xs, output)
    end
    return last(xs)
end

(p::SkipSequential)(x::Array{<:JuMPReal}) = apply(p, x)
