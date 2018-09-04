export SkipBlock

"""
$(TYPEDEF)

TODO (vtjeng)

## Fields:
$(FIELDS)
"""
@auto_hash_equals struct SkipBlock <: Layer
    layers::Array{<:Layer}
end

function Base.show(io::IO, p::SkipBlock)
    println(io, "SkipBlock")
    for (index, value) in enumerate(p.layers)
        println(io, "    ($index) $value")
    end
end

# TODO (vtjeng): better typing on xs
function apply(p::SkipBlock, xs::Array)
    num_layers = length(p.layers)
    inputs = xs[end-num_layers+1:end]
    outputs = map((f, x) -> f(x), p.layers, inputs)
    return outputs |> sum
end

(p::SkipBlock)(xs::Array) = apply(p, xs)