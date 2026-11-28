# Check every compiled affine coefficient against the independent native layer implementation.
# Arguments: network name (WK17a/RSL18a), output CSV.
using LinearAlgebra
BLAS.set_num_threads(1)
using CSV, MIPVerify
using MIPVerify.BackwardBounds: compile_stages

zero_bias(layer::MIPVerify.Conv2d) =
    MIPVerify.Conv2d(layer.filter, zero(layer.bias), layer.stride, layer.padding)
zero_bias(layer::MIPVerify.Linear) = MIPVerify.Linear(layer.matrix, zero(layer.bias))

function audit(network_name, output)
    network = MIPVerify.get_example_network_params("MNIST.$(network_name)_linf0.1_authors")
    shape = (1, 28, 28, 1)
    stages = compile_stages(network, shape)
    index = 0
    flatten = identity
    rows = NamedTuple[]
    for layer in network.layers
        if layer isa MIPVerify.Flatten
            flatten = layer
        elseif !(layer isa MIPVerify.ReLU)
            index += 1
            stage = stages[index]
            basis = zeros(shape)
            unbiased = zero_bias(layer)
            native_bias = layer(flatten(basis))
            @assert vec(native_bias) == stage.bias
            weights = Matrix(stage.weights)
            # A unit basis and zero biases copy one weight exactly, including padding/permutation.
            for pixel in eachindex(basis)
                basis[pixel] = 1
                @assert vec(unbiased(flatten(basis))) == view(weights, pixel, :) (index, pixel)
                basis[pixel] = 0
            end
            push!(
                rows,
                (;
                    network = network_name,
                    stage = index,
                    input_bases_checked = length(basis),
                    coefficients_checked = length(weights),
                    biases_checked = length(native_bias),
                    native_matches = true,
                ),
            )
            CSV.write(output, rows)
            println(network_name, " stage=", index, " coefficients=", length(weights))
            flush(stdout)
            shape = size(native_bias)
            flatten = identity
        end
    end
end

audit(ARGS...)
