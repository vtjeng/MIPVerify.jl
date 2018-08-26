export get_matrix_params, get_conv_params, get_example_network_params

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
   of the weights of the layer.
    
"""
function get_matrix_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{2, Int};
    matrix_name::String = "weight",
    bias_name::String = "bias")::Linear

    params = Linear(
        param_dict["$layer_name/$matrix_name"],
        squeeze(param_dict["$layer_name/$bias_name"], 1)
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
    of the weights of the layer.
    
"""
function get_conv_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{4, Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    expected_stride::Integer = 1
    )::Conv2d

    params = Conv2d(
        param_dict["$layer_name/$matrix_name"],
        squeeze(param_dict["$layer_name/$bias_name"], 1),
        expected_stride
    )

    check_size(params, expected_size)

    return params
end