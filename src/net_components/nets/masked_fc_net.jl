@auto_hash_equals struct MaskedFullyConnectedNetParameters <: NeuralNetParameters
    masked_fclayer_params::Array{MaskedFullyConnectedLayerParameters, 1}
    softmax_params::SoftmaxParameters
    UUID::String
end

function Base.show(io::IO, p::MaskedFullyConnectedNetParameters)
    fc_layer_text = (length(p.masked_fclayer_params) == 0) ? "\n    (none)" : join(string.([""; p.masked_fclayer_params]), "\n    ")
    softmax_text = string(
        "\n    ",
        string(p.softmax_params)
    )

    print(io,
        "masked fully-connected net $(p.UUID)",
        "\n  `masked_fclayer_params` [$(length(p.masked_fclayer_params))]:", fc_layer_text,
        "\n  `softmax_params`:", softmax_text
    )
end

(p::MaskedFullyConnectedNetParameters)(x::Array{<:JuMPReal, 4}) = (
    x |> MIPVerify.flatten |> p.masked_fclayer_params |> p.softmax_params
)