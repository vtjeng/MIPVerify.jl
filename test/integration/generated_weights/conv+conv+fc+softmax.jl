using MIPVerify: ConvolutionLayerParameters, SoftmaxParameters, FullyConnectedLayerParameters
using MIPVerify: StandardNeuralNetParameters
using MIPVerify: PerturbationParameters, AdditivePerturbationParameters, BlurPerturbationParameters
using MIPVerify.IntegrationTestHelpers: batch_test_adversarial_example
using Base.Test

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

conv1params = ConvolutionLayerParameters(
    rand(filter1_height, filter1_width, in1_channels, out1_channels)*2-1,
    rand(out1_channels)*2-1,
    strides1
)

conv2params = ConvolutionLayerParameters(
    rand(filter2_height, filter2_width, in2_channels, out2_channels)*2-1,
    rand(out2_channels)*2-1,
    strides2
)

fc1params = FullyConnectedLayerParameters(
    rand(A_width, A_height)*2-1,
    rand(A_height)*2-1
)

softmaxparams = SoftmaxParameters(
    rand(B_width, B_height)*2-1,
    rand(B_height)*2-1
)

nnparams = StandardNeuralNetParameters(
    [conv1params, conv2params], 
    [fc1params], 
    softmaxparams,
    "tests.integration.generated_weights.conv+conv+fc+softmax"
)

expected_objective_values::Dict{Int, Dict{PerturbationParameters, Dict{Real, Dict{Real, Float64}}}} = Dict(
    2 => Dict(
        AdditivePerturbationParameters() => Dict(
            1 => Dict(
                0.1 => 0.0,
                1 => 0.991616,
                1.5 => 3.97464,
                2 => NaN
            ),
            Inf => Dict(
                0.1 => 0.0,
                1 => 0.0953527,
                1.5 => 0.330050, # CBC: 0.3300592979616859, Gurobi: 0.3300495149023101
                2 => NaN
            )
        ),
        BlurPerturbationParameters((5, 5)) => Dict(
            1 => Dict(
                0.6 => 0.0,
                0.625 => 1.40955,
                0.65 => NaN,
            ),
            Inf => Dict(
                0.6 => 0.0,
                0.625 => 0.0570452,
                0.65 => NaN,
            )

        )
    )
)

batch_test_adversarial_example(nnparams, x0, expected_objective_values)

end