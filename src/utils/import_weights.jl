export get_matrix_params, get_conv_params, get_example_network_params
export convert_conv_filter_from_pytorch,
    convert_conv_filter_from_flux,
    convert_linear_weights_from_pytorch,
    convert_images_from_pytorch,
    convert_images_from_flux

"""
$(SIGNATURES)

Helper function to import the parameters for a layer carrying out matrix multiplication
    (e.g. fully connected layer / softmax layer) from `param_dict` as a
    [`Linear`](@ref) object.

The default format for the key is `'layer_name/weight'` and `'layer_name/bias'`;
    you can customize this by passing in the named arguments `matrix_name` and `bias_name`
    respectively. The expected parameter names will then be `'layer_name/matrix_name'`
    and `'layer_name/bias_name'`

# Arguments
* `param_dict::Dict{String}`: Dictionary mapping parameter names to array of weights
    / biases.
* `layer_name::String`: Identifies parameter in dictionary.
* `expected_size::NTuple{2, Int}`: Tuple of length 2 corresponding to the expected size
   of the weights of the layer, in `(in_features, out_features)` order.

"""
function get_matrix_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{2,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
)::Linear

    params = Linear(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
    )

    check_size(params, expected_size)

    return params
end

"""
$(SIGNATURES)

Helper function to import the parameters for a convolution layer from `param_dict` as a
    [`Conv2d`](@ref) object.

The default format for the key is `'layer_name/weight'` and `'layer_name/bias'`;
    you can customize this by passing in the named arguments `matrix_name` and `bias_name`
    respectively. The expected parameter names will then be `'layer_name/matrix_name'`
    and `'layer_name/bias_name'`

# Arguments
* `param_dict::Dict{String}`: Dictionary mapping parameter names to array of weights
    / biases.
* `layer_name::String`: Identifies parameter in dictionary.
* `expected_size::NTuple{4, Int}`: Tuple of length 4 corresponding to the expected size
    of the weights of the layer, in HWIO order:
    `(filter_height, filter_width, in_channels, out_channels)`.

"""
function get_conv_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{4,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    expected_stride::Integer = 1,
    padding::Padding = SamePadding(),
)::Conv2d

    params = Conv2d(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,
    )

    check_size(params, expected_size)

    return params
end

"""
    convert_conv_filter_from_pytorch(filter::Array{T,4}) -> Array{T,4}

Convert a 4D convolution filter from PyTorch's OIHW convention
`(out_channels, in_channels, height, width)` to MIPVerify's HWIO convention
`(height, width, in_channels, out_channels)`.
"""
function convert_conv_filter_from_pytorch(filter::Array{T,4}) where {T}
    permutedims(filter, (3, 4, 2, 1))
end

"""
    convert_conv_filter_from_flux(filter::Array{T,4}) -> Array{T,4}

Convert a 4D convolution filter from Flux.jl's WHIO convention
`(width, height, in_channels, out_channels)` to MIPVerify's HWIO convention
`(height, width, in_channels, out_channels)`.
"""
function convert_conv_filter_from_flux(filter::Array{T,4}) where {T}
    permutedims(filter, (2, 1, 3, 4))
end

"""
    convert_linear_weights_from_pytorch(matrix::Array{T,2}) -> Array{T,2}

Convert a weight matrix from PyTorch's `(out_features, in_features)` convention
to MIPVerify's `(in_features, out_features)` convention.
"""
function convert_linear_weights_from_pytorch(matrix::Array{T,2}) where {T}
    permutedims(matrix, (2, 1))
end

"""
    convert_images_from_pytorch(images::Array{T,4}) -> Array{T,4}

Convert a batch of images from PyTorch's NCHW convention
`(num_samples, channels, height, width)` to MIPVerify's NHWC convention
`(num_samples, height, width, channels)`.
"""
function convert_images_from_pytorch(images::Array{T,4}) where {T}
    permutedims(images, (1, 3, 4, 2))
end

"""
    convert_images_from_flux(images::Array{T,4}) -> Array{T,4}

Convert a batch of images from Flux.jl's WHCN convention
`(width, height, channels, num_samples)` to MIPVerify's NHWC convention
`(num_samples, height, width, channels)`.
"""
function convert_images_from_flux(images::Array{T,4}) where {T}
    permutedims(images, (4, 2, 1, 3))
end
