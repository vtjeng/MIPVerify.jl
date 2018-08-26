using Base.Test
using MIPVerify
using MIPVerify: BlurringPerturbationFamily
isdefined(:TestHelpers) || include("../../../TestHelpers.jl")
using TestHelpers: batch_test_adversarial_example

@testset "FC + Softmax" begin

### Parameters for neural net
batch = 1
in1_height = 8
in1_width = 8
in1_channels = 2

A_height = 5
A_width = batch*in1_height*in1_width*in1_channels

B_height = 3
B_width = A_height

srand(5)
input_size = (batch, in1_height, in1_width, in1_channels)
x0 = rand(input_size)

kernelf1 = rand(-10:10, A_width, A_height)
biasf1 = rand(-10:10, A_height)

kernelf2 = rand(-10:10, B_width, B_height)
biasf2 = rand(-10:10, B_height)

nn = Sequential([Flatten(4),
        Linear(kernelf1, biasf1), ReLU(),
        Linear(kernelf2, biasf2)
    ],
    "tests.integration.generated_weights.fc+softmax"
)

@testset "Blurring perturbations accept multiple input channels." begin
    pp_blur = BlurringPerturbationFamily((5, 5))

    expected_objective_values = Dict(
        (1, pp_blur, Inf, 0) => 0,
        (2, pp_blur, Inf, 0) => 0.1936444023676,
        (3, pp_blur, Inf, 0) => 0.4151904104813
    )

    batch_test_adversarial_example(nn, x0, expected_objective_values)
end

end