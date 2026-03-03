using Base.Cartesian

using JuMP

export Conv2d
export Padding, SamePadding, ValidPadding

struct SamePadding end
Base.show(io::IO, p::SamePadding) = print(io, "same")
struct ValidPadding end
Base.show(io::IO, p::ValidPadding) = print(io, "valid")

FixedPadding = Union{Int,Tuple{Int,Int},Tuple{Int,Int,Int,Int}}
Padding = Union{SamePadding,ValidPadding,FixedPadding}

"""
$(TYPEDEF)

Represents 2-D convolution operation.

`p(x)` is shorthand for [`conv2d(x, p)`](@ref) when `p` is an instance of
`Conv2d`.

## Dimension conventions (TensorFlow-style NHWC/HWIO):

This follows the conventions of
[`tf.nn.conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d):

- **Input**: `(batch, height, width, in_channels)` — NHWC format,
  matching TensorFlow's default `data_format="NHWC"`
- **Filter**: `(filter_height, filter_width, in_channels, out_channels)` — HWIO format,
  matching TensorFlow's filter shape for `tf.nn.conv2d`
- **Output**: `(batch, out_height, out_width, out_channels)` — NHWC format

To convert from PyTorch (OIHW filters, NCHW inputs), use
[`convert_conv_filter_from_pytorch`](@ref) and [`convert_images_from_pytorch`](@ref).

To convert from Flux.jl (WHIO filters, WHCN inputs), use
[`convert_conv_filter_from_flux`](@ref) and [`convert_images_from_flux`](@ref).

## Fields:
$(FIELDS)
"""
struct Conv2d{T<:JuMPReal,U<:JuMPReal,V<:Integer} <: Layer
    filter::Array{T,4}
    bias::Array{U,1}
    stride::V
    padding::Padding

    function Conv2d{T,U,V}(
        filter::Array{T,4},
        bias::Array{U,1},
        stride::V,
        padding::Padding,
    ) where {T<:JuMPReal,U<:JuMPReal,V<:Integer}
        (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(filter)
        bias_out_channels = length(bias)
        @assert(
            filter_out_channels == bias_out_channels,
            "For this convolution layer, number of output channels in filter, $filter_out_channels, does not match number of output channels in bias, $bias_out_channels."
        )
        return new(filter, bias, stride, padding)
    end

end

function Conv2d(
    filter::Array{T,4},
    bias::Array{U,1},
    stride::V,
    padding::Padding,
) where {T<:JuMPReal,U<:JuMPReal,V<:Integer}
    Conv2d{T,U,V}(filter, bias, stride, padding)
end

function Conv2d(
    filter::Array{T,4},
    bias::Array{U,1},
    stride::V,
) where {T<:JuMPReal,U<:JuMPReal,V<:Integer}
    Conv2d{T,U,V}(filter, bias, stride, SamePadding())
end

function Conv2d(filter::Array{T,4}, bias::Array{U,1}) where {T<:JuMPReal,U<:JuMPReal}
    Conv2d(filter, bias, 1, SamePadding())
end

"""
$(SIGNATURES)

Convenience function to create a [`Conv2d`](@ref) struct with the specified filter
and zero bias.
"""
function Conv2d(filter::Array{T,4}) where {T<:JuMPReal}
    bias_out_channels::Int = size(filter)[4]
    bias = zeros(bias_out_channels)
    Conv2d(filter, bias)
end

function check_size(params::Conv2d, sizes::NTuple{4,Int})::Nothing
    check_size(params.filter, sizes)
    check_size(params.bias, (sizes[end],))
end

function Base.show(io::IO, p::Conv2d)
    (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(p.filter)
    stride = p.stride
    padding = p.padding
    print(
        io,
        "Conv2d($filter_in_channels, $filter_out_channels, kernel_size=($(filter_height), $(filter_width)), stride=($(stride), $(stride)), padding=$(padding))",
    )
end

# TODO (vtjeng): Figure out how to actually mutate the underlying value of s
# OR avoid all this confusion
function add_to_expression!(s::Real, input_val::Real, filter_val::Real)
    return s + input_val * filter_val
end

function add_to_expression!(s::JuMP.GenericAffExpr, input_val, filter_val)
    return JuMP.add_to_expression!(s, input_val, filter_val)
end

function compute_output_parameters(
    in_height::Int,
    in_width::Int,
    filter_height::Int,
    filter_width::Int,
    stride::Int,
    padding::FixedPadding,
)::Tuple{NTuple{2,Int},NTuple{2,Int}}
    (top_padding, bottom_padding, left_padding, right_padding) = compute_padding_values(padding)
    out_height_raw = (in_height + top_padding + bottom_padding - filter_height) / stride
    out_height = round(Int, out_height_raw, RoundDown) + 1
    out_width_raw = (in_width + left_padding + right_padding - filter_width) / stride
    out_width = round(Int, out_width_raw, RoundDown) + 1

    output_size = (out_height, out_width)
    filter_offset = (top_padding, left_padding)
    return (output_size, filter_offset)
end

function compute_output_parameters(
    in_height::Int,
    in_width::Int,
    filter_height::Int,
    filter_width::Int,
    stride::Int,
    padding::SamePadding,
)::Tuple{NTuple{2,Int},NTuple{2,Int}}
    out_height = round(Int, in_height / stride, RoundUp)
    out_width = round(Int, in_width / stride, RoundUp)
    pad_along_height = max((out_height - 1) * stride + filter_height - in_height, 0)
    pad_along_width = max((out_width - 1) * stride + filter_width - in_width, 0)
    filter_height_offset = round(Int, pad_along_height / 2, RoundDown)
    filter_width_offset = round(Int, pad_along_width / 2, RoundDown)

    output_size = (out_height, out_width)
    filter_offset = (filter_height_offset, filter_width_offset)
    return (output_size, filter_offset)
end

function compute_output_parameters(
    in_height::Int,
    in_width::Int,
    filter_height::Int,
    filter_width::Int,
    stride::Int,
    padding::ValidPadding,
)::Tuple{NTuple{2,Int},NTuple{2,Int}}
    out_height = round(Int, (in_height + 1 - filter_height) / stride, RoundUp)
    out_width = round(Int, (in_width + 1 - filter_width) / stride, RoundUp)
    return ((out_height, out_width), (0, 0))
end

function compute_padding_values(padding::Int)::NTuple{4,Int}
    return (padding, padding, padding, padding)
end

function compute_padding_values(padding::NTuple{2,Int})::NTuple{4,Int}
    (y_padding, x_padding) = padding
    return (y_padding, y_padding, x_padding, x_padding)
end

function compute_padding_values(padding::NTuple{4,Int})::NTuple{4,Int}
    return padding
end

"""
$(SIGNATURES)

Computes the result of convolving `input` with the `filter` and `bias` stored in `params`.

Mirrors [`tf.nn.conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
from TensorFlow, with `strides = [1, params.stride, params.stride, 1]`.

## Dimension conventions:
- **Input**: `(batch, height, width, in_channels)` — NHWC format
- **Filter**: `(filter_height, filter_width, in_channels, out_channels)` — HWIO format
- **Output**: `(batch, out_height, out_width, out_channels)` — NHWC format

## Padding:
- `SamePadding()`: TensorFlow-style `SAME` padding is used, so output spatial size is
  `(ceil(input_height / stride), ceil(input_width / stride))`.
- `ValidPadding()`: No padding is added.
- Fixed padding, specified as:
  - A single integer, interpreted as padding for both axes
  - A tuple of two integers, interpreted as `(y_padding, x_padding)`
  - A tuple of four integers, interpreted as `(top, bottom, left, right)`

# Throws
* AssertionError if `input` and `filter` are not compatible.
"""
function conv2d(input::Array{T,4}, params::Conv2d{U,V}) where {T<:JuMPReal,U<:JuMPReal,V<:JuMPReal}

    if T <: JuMPLinearType || U <: JuMPLinearType || V <: JuMPLinearType
        info(MIPVerify.LOGGER, "Applying $(params) ... ")
    end
    filter = params.filter
    stride = params.stride
    padding = params.padding

    (batch, in_height, in_width, input_in_channels) = size(input)
    (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(filter)

    @assert(
        input_in_channels == filter_in_channels,
        "Number of channels in input, $input_in_channels, does not match number of channels, $filter_in_channels, that filters operate on."
    )

    # Considered using offset arrays here, but could not get it working.
    ((out_height, out_width), (filter_height_offset, filter_width_offset)) =
        compute_output_parameters(in_height, in_width, filter_height, filter_width, stride, padding)
    output_size = (batch, out_height, out_width, filter_out_channels)

    W = Base.promote_op(+, V, Base.promote_op(*, T, U))
    output = Array{W}(undef, output_size)

    @nloops 4 i output begin
        (@nref 4 output i) = params.bias[i_4]
        @nloops 3 j filter begin
            x = (i_2 - 1) * stride + j_1 - filter_height_offset
            y = (i_3 - 1) * stride + j_2 - filter_width_offset
            input_index = (i_1, x, y, j_3)
            if checkbounds(Bool, input, input_index...)
                # Effectively zero-padding the input.
                (@nref 4 output i) = add_to_expression!(
                    (@nref 4 output i),
                    input[input_index...],
                    filter[j_1, j_2, j_3, i_4],
                )
            end
        end
    end

    return output
end

(p::Conv2d)(x::Array{<:JuMPReal,4}) = conv2d(x, p)
