using Base.Test

using MIPVerify: ConvolutionLayerParameters, FullyConnectedLayerParameters
using MIPVerify: SoftmaxParameters, StandardNeuralNetParameters

@testset "StandardNeuralNetParameters" begin
    
    filter_out_channels = 5
    filter = rand(3, 3, 1, filter_out_channels)
    bias = rand(filter_out_channels)
    strides = (1, 2, 2, 1)
    c1params = ConvolutionLayerParameters(filter, bias, strides)

    A_height = 40
    A_width = filter_out_channels*14*14
            
    B_height = 20
    B_width = A_height
            
    C_height = 10
    C_width = B_height
    
    fc1params = FullyConnectedLayerParameters(rand(A_width, A_height), rand(A_height))
    fc2params = FullyConnectedLayerParameters(rand(B_width, B_height), rand(B_height))
    softmaxparams = SoftmaxParameters(rand(C_width, C_height), rand(C_height))

    nnparams = StandardNeuralNetParameters(
        [c1params], 
        [fc1params, fc2params], 
        softmaxparams,
        "testnet"
    )
    @testset "Base.show" begin
        io = IOBuffer()
        Base.show(io, nnparams)
        @test String(take!(io)) == """
            convolutional neural net testnet
              `convlayer_params` [1]:
                convolution layer. applies 5 3x3 filters with stride 1, followed by max pooling with a 2x2 filter and a stride of (2, 2), and a ReLU activation function.
              `fclayer_params` [2]:
                fully connected layer with 980 inputs and 40 output units, and a ReLU activation function.
                fully connected layer with 40 inputs and 20 output units, and a ReLU activation function.
              `softmax_params`:
                softmax layer with 20 inputs and 10 output units."""
    end
end