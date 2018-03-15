using Base.Test
using MIPVerify
using MIPVerify: UnrestrictedPerturbationFamily, BlurringPerturbationFamily
using MIPVerify: LInfNormBoundedPerturbationFamily
isdefined(:TestHelpers) || include("../../../TestHelpers.jl")
using TestHelpers: batch_test_adversarial_example

@testset "Conv + Conv + FC + Softmax" begin

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

in2_height = pooled1_height
in2_width = pooled1_width
stride2_height = 1
stride2_width = 1
strides2 = (1, stride2_height, stride2_width, 1)
pooled2_height = round(Int, in2_height/stride2_height, RoundUp)
pooled2_width = round(Int, in2_width/stride2_width, RoundUp)
in2_channels = out1_channels
filter2_height = 2
filter2_width = 2
out2_channels = 2

A_height = 5
A_width = pooled2_height*pooled2_width*out2_channels

B_height = 3
B_width = A_height

### Choosing data to be used
srand(5)
input_size = (batch, in1_height, in1_width, in1_channels)
x0 = rand(input_size)

kernelc1 = rand(filter1_height, filter1_width, in1_channels, out1_channels)*2-1
biasc1 = rand(out1_channels)*2-1

kernelc2 = rand(filter2_height, filter2_width, in2_channels, out2_channels)*2-1
biasc2 = rand(out2_channels)*2-1

kernelf1 = rand(A_width, A_height)*2-1
biasf1 = rand(A_height)*2-1

kernelf2 = rand(B_width, B_height)*2-1
biasf2 = rand(B_height)*2-1

nn = Sequential(
    [
        Conv2d(kernelc1, biasc1), MaxPool(strides1), ReLU(),
        Conv2d(kernelc2, biasc2), MaxPool(strides2), ReLU(),
        Flatten(4),
        Linear(kernelf1, biasf1), ReLU(),
        Linear(kernelf2, biasf2)
    ],
    "tests.integration.generated_weights.conv+conv+fc+softmax"
)

pp_blur = BlurringPerturbationFamily((5, 5))
pp_unrestricted = UnrestrictedPerturbationFamily()

expected_objective_values = Dict(
    (2, pp_unrestricted, 1, 0.1) => 0.0,
    (2, pp_unrestricted, 1, 1) => 0.991616,
    (2, pp_unrestricted, 1, 2) => NaN,
    (2, pp_unrestricted, Inf, 0.1) => 0.0,
    (2, pp_unrestricted, Inf, 1) => 0.0953527,
    (2, LInfNormBoundedPerturbationFamily(0.09), Inf, 1) => NaN,
    (2, LInfNormBoundedPerturbationFamily(0.0954), Inf, 1) => 0.0953527,
    (2, LInfNormBoundedPerturbationFamily(0.096), Inf, 1) => 0.0953527,
    (2, pp_blur, 1, 0.6) => 0.0,
    (2, pp_blur, 1, 0.625) => 1.40955,
    (2, pp_blur, 1, 0.65) => NaN,
    (2, pp_blur, Inf, 0.6) => 0.0,
    (2, pp_blur, Inf, 0.625) => 0.0570452
)

batch_test_adversarial_example(nn, x0, expected_objective_values)

end