export get_matrix_params, get_conv_params, get_example_network_params

"""
$(SIGNATURES)

Helper function to import the parameters for a layer carrying out matrix multiplication 
    (e.g. fully connected layer / softmax layer) from `param_dict` as a
    [`MatrixMultiplicationParameters`](@ref) object.

The default format for parameter names is `'layer_name/weight'` and `'layer_name/bias'`; 
    you can customize this by passing in the named arguments `matrix_name` and `bias_name`
    respectively.

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
    bias_name::String = "bias")::MatrixMultiplicationParameters

    params = MatrixMultiplicationParameters(
        param_dict["$layer_name/$matrix_name"],
        squeeze(param_dict["$layer_name/$bias_name"], 1)
    )

    check_size(params, expected_size)

    return params
end

"""
$(SIGNATURES)

Helper function to import the parameters for a convolution layer from `param_dict` as a
    [`Conv2DParameters`](@ref) object.

The default format for parameter names is `'layer_name/weight'` and `'layer_name/bias'`; 
    you can customize this by passing in the named arguments `matrix_name` and `bias_name`
    respectively.
    
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
    bias_name::String = "bias")::Conv2DParameters

    params = Conv2DParameters(
        param_dict["$layer_name/$matrix_name"],
        squeeze(param_dict["$layer_name/$bias_name"], 1)
    )

    check_size(params, expected_size)

    return params
end

"""
$(SIGNATURES)

Makes named example neural networks available as a [`NeuralNetParameters`](@ref) object.

# Arguments
* `name::String`: Name of example neural network. Options:
    * `'MNIST.n1'`: MNIST classification. Two fully connected layers with 40 and 20
        units, and softmax layer with 10 units. No adversarial training.
"""
function get_example_network_params(name::String)::NeuralNetParameters
    if name == "MNIST.n1"
        in1_height = 28
        in1_width = 28
        
        A_height = 40
        A_width = in1_height*in1_width
        
        B_height = 20
        B_width = A_height
        
        C_height = 10
        C_width = B_height
        
        param_dict = prep_data_file(joinpath("weights", "mnist"), "n1.mat") |> matread
        fc1params = get_matrix_params(param_dict, "fc1", (A_width, A_height)) |> FullyConnectedLayerParameters
        fc2params = get_matrix_params(param_dict, "fc2", (B_width, B_height)) |> FullyConnectedLayerParameters
        softmaxparams = get_matrix_params(param_dict, "logits", (C_width, C_height)) |> SoftmaxParameters
        
        nnparams = StandardNeuralNetParameters(
            ConvolutionLayerParameters[], 
            [fc1params, fc2params], 
            softmaxparams,
            name
        )
        return nnparams
    else
        throw(DomainError("No example network named $name."))
    end
end