export get_matrix_params, get_conv_params, get_example_network_params

"""
$(SIGNATURES)

Helper function to import the parameters for a layer carrying out matrix multiplication 
    (e.g. fully connected layer / softmax layer) from `param_dict` as a
    [`Linear`](@ref) object.

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
    bias_name::String = "bias",
    expected_stride::Int = 1
    )::Conv2d

    params = Conv2d(
        param_dict["$layer_name/$matrix_name"],
        squeeze(param_dict["$layer_name/$bias_name"], 1),
        expected_stride
    )

    check_size(params, expected_size)

    return params
end

"""
$(SIGNATURES)

Makes named example neural networks available as a [`NeuralNet`](@ref) object.

# Arguments
* `name::String`: Name of example neural network. Options:
    * `'MNIST.n1'`: 
        * Architecture: Two fully connected layers with 40 and 20 units, and 
          softmax layer with 10 units. 
        * Training: Trained regularly with no attempt to increase robustness.
    * `'MNIST.WK17a_linf0.1_authors'`. 
        * Architecture: Two convolutional layers (stride length 2) with 16 and 
          32 filters respectively (size 4 × 4 in both layers), followed by a 
          fully-connected layer with 100 units. 
        * Training: Network trained to be robust to attacks with \$l_\\infty\$ norm
          at most 0.1 via method in [Provable defenses against adversarial examples 
          via the convex outer adversarial polytope](https://arxiv.org/abs/1711.00851). 
          Is MNIST network for which results are reported in that paper.
"""
function get_example_network_params(name::String)::NeuralNet
    if name == "MNIST.n1"       
        param_dict = prep_data_file(joinpath("weights", "mnist"), "n1.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 40))
        fc2 = get_matrix_params(param_dict, "fc2", (40, 20))
        logits = get_matrix_params(param_dict, "logits", (20, 10))
        
        nn = Sequential([
            Flatten(4),
            fc1, ReLU(),
            fc2, ReLU(),
            logits], name)
        return nn
    elseif name == "MNIST.WK17a_linf0.1_authors"
        param_dict = prep_data_file(joinpath("weights", "mnist", "WK17a", "linf0.1"), "master_seed_1_epochs_100.mat") |> matread
        conv1 = get_conv_params(param_dict, "conv1", (4, 4, 1, 16), expected_stride = 2)
        conv2 = get_conv_params(param_dict, "conv2", (4, 4, 16, 32), expected_stride = 2)
        fc1 = get_matrix_params(param_dict, "fc1", (1568, 100))
        logits = get_matrix_params(param_dict, "logits", (100, 10))
        
        nn = Sequential([
            conv1, ReLU(),
            conv2, ReLU(),
            Flatten([1, 3, 2, 4]),
            fc1, ReLU(),
            logits], name)
        return nn
    else
        throw(DomainError("No example network named $name."))
    end
end

# TODO (vtjeng): Add mnist networks Ragunathan/Steinhardt/Liang.
# TODO (vtjeng): Make network naming case insensitive.