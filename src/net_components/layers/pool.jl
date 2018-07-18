export Pool, MaxPool

"""
$(TYPEDEF)

Represents a pooling operation.

`p(x)` is shorthand for [`pool(x, p)`](@ref) when `p` is an instance of `Pool`.

## Fields:
$(FIELDS)
"""
struct Pool{N} <: Layer
    strides::NTuple{N, Int}
    pooling_function::Function
end

function Base.show(io::IO, p::Pool)
    (_, stride_height, stride_width, _) = p.strides
    function_display_name = Dict(
        MIPVerify.maximum => "max",
        Base.mean => "average",
    )
    print(io,
        "$(function_display_name[p.pooling_function]) pooling with a $(stride_height)x$(stride_width) filter and a stride of ($stride_height, $stride_width)"
    )
end

Base.hash(a::Pool, h::UInt) = hash(a.strides, hash(string(a.pooling_function), hash(:Pool, h)))

"""
$(SIGNATURES)

Convenience function to create a [`Pool`](@ref) struct for max-pooling.
"""
function MaxPool(strides::NTuple{N, Int}) where {N}
    Pool(strides, MIPVerify.maximum)
end

function AveragePool(strides::NTuple{N, Int}) where {N}
    # TODO (vtjeng): support average pooling across variables.
    Pool(strides, Base.mean)
end

"""
$(SIGNATURES)

For pooling operations on an array where a given element in the output array
corresponds to equal-sized blocks in the input array, returns (for a given
dimension) the index range in the input array corresponding to a particular
index `output_index` in the output array.

Returns an empty array if the `output_index` does not correspond to any input
indices.

# Arguments
* `stride::Integer`: the size of the operating blocks along the active
     dimension.

"""
function getsliceindex(input_array_size::Integer, stride::Integer, output_index::Integer)::Array{Int, 1}
    parent_start_index = (output_index-1)*stride+1
    parent_end_index = min((output_index)*stride, input_array_size)
    if parent_start_index > parent_end_index
        return []
    else
        return parent_start_index:parent_end_index
    end
end

"""
$(SIGNATURES)

For pooling operations on an array, returns a view of the parent array
corresponding to the `output_index` in the output array.
"""
function getpoolview(input_array::AbstractArray{T, N}, strides::NTuple{N, Int}, output_index::NTuple{N, Int})::SubArray{T, N} where {T, N}
    it = zip(size(input_array), strides, output_index)
    input_index_range = map(x -> getsliceindex(x...), it)
    return view(input_array, input_index_range...)
end

"""
$(SIGNATURES)

For pooling operations on an array, returns the expected size of the output
array.
"""
function getoutputsize(input_array::AbstractArray{T, N}, strides::NTuple{N, Int})::NTuple{N, Int} where {T, N}
    output_size = ((x, y) -> round(Int, x/y, RoundUp)).(size(input_array), strides)
    return output_size
end

"""
$(SIGNATURES)

Returns output from applying `f` to subarrays of `input_array`, with the windows
determined by the `strides`.
"""
function poolmap(f::Function, input_array::AbstractArray{T, N}, strides::NTuple{N, Int}) where {T, N}
    output_size = getoutputsize(input_array, strides)
    output_indices = collect(CartesianRange(output_size))
    return ((I) -> f(getpoolview(input_array, strides, I.I))).(output_indices)
end

"""
$(SIGNATURES)

Computes the result of applying the pooling function `params.pooling_function` to 
non-overlapping cells of `input` with sizes specified in `params.strides`.
"""
function pool(
    input::AbstractArray{T, N},
    params::Pool{N}) where {T<:JuMPReal, N}
    poolmap(params.pooling_function, input, params.strides)
end

(p::Pool)(x::Array{<:Real}) = MIPVerify.pool(x, p)
(p::Pool)(x::Array{<:JuMPLinearType}) = (info(MIPVerify.LOGGER, "Specifying $p ... "); MIPVerify.pool(x, p))