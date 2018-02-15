@auto_hash_equals struct StandardNeuralNetParameters <: NeuralNetParameters
    convlayer_params::Array{ConvolutionLayerParameters, 1}
    fclayer_params::Array{FullyConnectedLayerParameters, 1}
    softmax_params::SoftmaxParameters
    UUID::String
end

function Base.show(io::IO, p::StandardNeuralNetParameters)
    convolutional_layer_text = (length(p.convlayer_params) == 0) ? "\n    (none)" : join(string.([""; p.convlayer_params]), "\n    ")
    fc_layer_text = (length(p.fclayer_params) == 0) ? "\n    (none)" : join(string.([""; p.fclayer_params]), "\n    ")
    softmax_text = string(
        "\n    ",
        string(p.softmax_params)
    )

    print(io,
        "convolutional neural net $(p.UUID)",
        "\n  `convlayer_params` [$(length(p.convlayer_params))]:", convolutional_layer_text,
        "\n  `fclayer_params` [$(length(p.fclayer_params))]:", fc_layer_text,
        "\n  `softmax_params`:", softmax_text
    )
end

(p::StandardNeuralNetParameters)(x::Array{<:JuMPReal, 4}) = (
    x |> p.convlayer_params |> MIPVerify.flatten |> p.fclayer_params |> p.softmax_params
)