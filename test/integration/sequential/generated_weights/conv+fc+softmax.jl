using Base.Test
using MIPVerify
using MIPVerify: UnrestrictedPerturbationFamily, BlurringPerturbationFamily
using MIPVerify: LInfNormBoundedPerturbationFamily
isdefined(:TestHelpers) || include("../../../TestHelpers.jl")
using TestHelpers: batch_test_adversarial_example

@testset "Conv + FC + Softmax" begin

### Parameters for neural net
batch = 1
in1_height = 8
in1_width = 8
stride1_height = 2
stride1_width = 2
strides1 = (1, stride1_height, stride1_width, 1)
pooled1_height = round(Int, in1_height/stride1_height, RoundUp)
pooled1_width = round(Int, in1_width/stride1_width, RoundUp)
in1_channels = 1
filter1_height = 2
filter1_width = 2
out1_channels = 2

A_height = 5
A_width = pooled1_height*pooled1_width*out1_channels

B_height = 3
B_width = A_height

srand(5)
input_size = (batch, in1_height, in1_width, in1_channels)
x0 = rand(input_size)

kernelc1 = rand(filter1_height, filter1_width, in1_channels, out1_channels)*2-1
biasc1 = rand(out1_channels)*2-1

kernelf1 = rand(-10:10, A_width, A_height)
biasf1 = rand(-10:10, A_height)

kernelf2 = rand(B_width, B_height)*2-1
biasf2 = rand(B_height)*2-1

nn = Sequential(
    [
        Conv2d(kernelc1, biasc1), MaxPool(strides1), ReLU(),
        Flatten(4),
        Linear(kernelf1, biasf1), ReLU(),
        Linear(kernelf2, biasf2)
    ],
    "tests.integration.generated_weights.conv+fc+softmax"
)

pp_blur = BlurringPerturbationFamily((5, 5))
pp_unrestricted = UnrestrictedPerturbationFamily()

expected_objective_values = Dict(
    (1, pp_blur, 1, 0) => 0,
    (1, pp_blur, Inf, 0) => 0,
    (2, pp_unrestricted, 1, 0) => 2.98266,
    (2, pp_unrestricted, 1, 0.1) => 3.03465,
    (2, pp_unrestricted, Inf, 0) => 0.235631,
    (2, LInfNormBoundedPerturbationFamily(0.23), Inf, 0) => NaN,
    (2, LInfNormBoundedPerturbationFamily(0.236), Inf, 0) => 0.235631,
    (2, pp_unrestricted, Inf, 0.1) => 0.240124,
    (2, pp_blur, 1, 0) => NaN,
    (2, pp_blur, Inf, 0) => NaN,
    (3, pp_unrestricted, Inf, 0) => 0.00288325,
    (3, pp_unrestricted, Inf, 1) => 0.0110296,
    (3, pp_blur, 1, 0) => 0.261483,
    (3, pp_blur, 1, 10) => NaN,
    (3, pp_blur, Inf, 0) => 0.0105534,
    (3, pp_blur, Inf, 10) => NaN,
    ([2, 3], pp_unrestricted, Inf, 0) => 0.00288325,
    ([2, 3], pp_unrestricted, Inf, 1) => 0.0110296,
    ([2, 3], pp_blur, 1, 0) => 0.261483,
    ([2, 3], pp_blur, Inf, 0) => 0.0105534
)

if Pkg.installed("Gurobi") == nothing
    # Skip some tests if Gurobi is not installed.
    selected_test_keys = [
        (2, pp_unrestricted, Inf, 0),
        (2, LInfNormBoundedPerturbationFamily(0.23), Inf, 0),
        (2, LInfNormBoundedPerturbationFamily(0.236), Inf, 0),
        (3, pp_blur, 1, 0),
        ([2, 3], pp_blur, 1, 0)
    ]
    expected_objective_values = filter((k, v) -> k in selected_test_keys, expected_objective_values)
end

batch_test_adversarial_example(nn, x0, expected_objective_values)

end