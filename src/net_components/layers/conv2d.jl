using JuMP

@auto_hash_equals struct Conv2DParameters{T<:JuMPReal, U<:JuMPReal} <: LayerParameters
    filter::Array{T, 4}
    bias::Array{U, 1}

    function Conv2DParameters{T, U}(filter::Array{T, 4}, bias::Array{U, 1}) where {T<:JuMPReal, U<:JuMPReal}
        (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(filter)
        bias_out_channels = length(bias)
        @assert(
            filter_out_channels == bias_out_channels,
            "For the convolution layer, number of output channels in filter, $filter_out_channels, does not match number of output channels in bias, $bias_out_channels."
        )
        return new(filter, bias)
    end

end

function Conv2DParameters(filter::Array{T, 4}, bias::Array{U, 1}) where {T<:JuMPReal, U<:JuMPReal}
    Conv2DParameters{T, U}(filter, bias)
end

function Conv2DParameters(filter::Array{T, 4}) where {T<:JuMPReal}
    bias_out_channels::Int = size(filter)[4]
    bias = zeros(bias_out_channels)
    Conv2DParameters(filter, bias)
end

function check_size(params::Conv2DParameters, sizes::NTuple{4, Int})::Void
    check_size(params.filter, sizes)
    check_size(params.bias, (sizes[end], ))
end

function Base.show(io::IO, p::Conv2DParameters)
    (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(p.filter)
    print(io,
        "applies $filter_out_channels $(filter_height)x$(filter_width) filters"
    )
end



function increment!(s::Real, input_val::Real, filter_val::Real)
    return s + input_val*filter_val
end

function increment!(s::JuMP.AffExpr, input_val::JuMP.AffExpr, filter_val::Real)
    append!(s, input_val*filter_val)
    return s
end

function increment!(s::JuMP.AffExpr, input_val::JuMP.Variable, filter_val::Real)
    push!(s, Float64(filter_val), input_val)
end

function increment!(s::JuMP.AffExpr, input_val::Real, filter_val::JuMP.AffExpr)
    append!(s, filter_val*input_val)
    return s
end

function increment!(s::JuMP.AffExpr, input_val::Real, filter_val::JuMP.Variable)
    push!(s, Float64(input_val), filter_val)
end



"""
Computes a 2D-convolution given 4-D `input` and `filter` tensors.

Mirrors `tf.nn.conv2d` from `tensorflow` package, with `strides` = [1, 1, 1, 1],
 `padding` = 'SAME'.

 # Throws
 * AssertionError if input and filter are not compatible.
"""
function conv2d(
    input::Array{T, 4},
    params::Conv2DParameters{U, V}) where {T<:JuMPReal, U<:JuMPReal, V<:JuMPReal}

    if T<:JuMP.AbstractJuMPScalar || U<:JuMP.AbstractJuMPScalar || V<:JuMP.AbstractJuMPScalar
        notice(MIPVerify.LOGGER, "Specifying conv2d constraints ... ")
    end
    filter = params.filter

    (batch, in_height, in_width, input_in_channels) = size(input)
    (filter_height, filter_width, filter_in_channels, filter_out_channels) = size(filter)
    
    @assert(
        input_in_channels == filter_in_channels, 
        "Number of channels in input, $input_in_channels, does not match number of channels, $filter_in_channels, that filters operate on."
    )
    
    output_size = (batch, in_height, in_width, filter_out_channels)

    # Considered using offset arrays here, but could not get it working.

    # Calculating appropriate offsets so that center of kernel is matched with
    # cell at which correlation is being calculated. Note that tensorflow
    # chooses a specific convention for a dimension with even size which we
    # replicate here.
    filter_height_offset = round(Int, filter_height/2, RoundUp)
    filter_width_offset = round(Int, filter_width/2, RoundUp)
    W = Base.promote_op(+, V, Base.promote_op(*, T, U))
    output = Array{W}(output_size)

    @nloops 4 i output begin
        s::W = 0
        @nloops 4 j filter begin
            if i_4 == j_4
                x = i_2 + j_1 - filter_height_offset
                y = i_3 + j_2 - filter_width_offset
                if x > 0 && y > 0 && x<=in_height && y<=in_width
                    # Doing bounds check to make sure that we stay within bounds
                    # for input. This effectively zero-pads the input.
                    # TODO: Use default checkbounds function here instead?
                    s = increment!(s, input[i_1, x, y, j_3], filter[j_1, j_2, j_3, j_4])
                end
            end
        end
        s += params.bias[i_4]
        (@nref 4 output i) = s
    end

    return output
end

(p::Conv2DParameters)(x::Array{<:JuMPReal, 4}) = conv2d(x, p)