export SkipBlock

"""
$(TYPEDEF)

A layer that implements a skip connection pattern, commonly used in residual networks (ResNets).

A `SkipBlock` takes multiple input tensors and applies a corresponding transformation layer to each input.
The outputs of these transformations are then combined through element-wise addition.

When used within a [`SkipSequential`](@ref), a `SkipBlock` processes inputs from the most recent layers
in the network. If the block has `n` layers, it will take the outputs from the last `n` layers as input,
in order. For example, with a 2-layer `SkipBlock`:
- The second layer processes the output from the immediately preceding layer
- The first layer processes the output from two layers back

Typically used in conjunction with [`SkipSequential`](@ref) to create networks with skip connections,
where the outputs from earlier layers can bypass intermediate layers and be combined
with later layer outputs.

Example:
```julia
skip_block = SkipBlock([
    Linear(input_size_1, output_size),  # Transform from one path
    Linear(input_size_2, output_size)   # Transform from another path
])
```

## Fields:
$(FIELDS)
"""
struct SkipBlock <: Layer
    layers::Array{<:Layer,1}
end

function Base.show(io::IO, p::SkipBlock)
    println(io, "SkipBlock")
    for (index, value) in enumerate(p.layers)
        println(io, "    ($index) $value")
    end
end

function apply(p::SkipBlock, xs::AbstractVector{<:AbstractArray{<:JuMPReal}})
    num_layers = length(p.layers)
    @assert num_layers > 0 "SkipBlock must contain at least one layer."
    inputs = xs[end-num_layers+1:end]
    outputs = map((f, x) -> f(x), p.layers, inputs)

    reference_size = size(outputs[1])
    if any(size(output) != reference_size for output in outputs[2:end])
        throw(
            DimensionMismatch(
                "All layer outputs in a SkipBlock must have the same size. Got sizes $(map(size, outputs)).",
            ),
        )
    end
    return reduce((x1, x2) -> x1 .+ x2, outputs)
end

(p::SkipBlock)(xs::AbstractVector{<:AbstractArray{<:JuMPReal}}) = apply(p, xs)
