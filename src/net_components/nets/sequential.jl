export Sequential

"""
$(TYPEDEF)

Represents a sequential (feed-forward) neural net, with `layers` ordered from input
to output.

## Fields:
$(FIELDS)
"""
@auto_hash_equals struct Sequential <: NeuralNet
    layers::Array{Layer, 1}
    UUID::String
end

function Base.show(io::IO, p::Sequential)

    println(io, "sequential net $(p.UUID)")
    for (index, value) in enumerate(p.layers)
        println(io, "  ($index) $value")
    end
end

(p::Sequential)(x::Array{<:JuMPReal}) = x |> p.layers
