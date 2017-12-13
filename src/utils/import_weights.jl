function get_matrix_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{2, Int};
    matrix_name::String = "weight",
    bias_name::String = "bias")

    params = MatrixMultiplicationParameters(
        param_dict["$layer_name/$matrix_name"],
        squeeze(param_dict["$layer_name/$bias_name"], 1)
    )

    check_size(params, expected_size)

    return params
end

function get_conv_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{4, Int};
    matrix_name::String = "weight",
    bias_name::String = "bias")

    params = Conv2DParameters(
        param_dict["$layer_name/$matrix_name"],
        squeeze(param_dict["$layer_name/$bias_name"], 1)
    )

    check_size(params, expected_size)

    return params
end

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
        
        param_dict = matread(joinpath("deps", "weights", "mnist", "n1.mat"))
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
        throw(ArgumentError("No example network named $name."))
    end
end