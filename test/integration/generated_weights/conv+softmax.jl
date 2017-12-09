using JuMP

using MIPVerify: find_adversarial_example, ConvolutionLayerParameters, SoftmaxParameters, StandardNeuralNetParameters, FullyConnectedLayerParameters, BlurPerturbationParameters
using Base.Test

@testset "Conv + Softmax" begin

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
out1_channels = 3

B_height = 3
B_width = pooled1_height*pooled1_width*out1_channels

### Choosing data to be used
srand(5)
x0 = rand(batch, in1_height, in1_width, in1_channels)

conv1params = ConvolutionLayerParameters(
    rand(filter1_height, filter1_width, in1_channels, out1_channels)*2-1,
    rand(out1_channels)*2-1,
    strides1
)

softmaxparams = SoftmaxParameters(
    rand(B_width, B_height)*2-1,
    rand(B_height)*2-1
)
nnparams = StandardNeuralNetParameters(
    [conv1params], 
    [],
    softmaxparams,
    "g02"
)

@testset "Additive Adversarial Example" begin
    d = find_adversarial_example(nnparams, x0, 1, tolerance=1.0, norm_type = 1, rebuild=true)
    @test getobjectivevalue(d[:Model]) ≈ 2.344824299053464    
    # Gurobi : 2.344824299053464
    # Cbc    : 2.3448242990534602
    d = find_adversarial_example(nnparams, x0, 1, tolerance=1.0, norm_type = typemax(Int), rebuild=true)
    @test getobjectivevalue(d[:Model]) ≈ 0.15628022275388148
    # Gurobi : 0.15628022275388148
end

end