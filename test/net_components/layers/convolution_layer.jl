using Base.Test
using MIPVerify: ConvolutionLayerParameters, check_size

@testset "convolution_layer.jl" begin

@testset "ConvolutionLayerParameters" begin
    filter_out_channels = 5
    filter = rand(3, 3, 1, filter_out_channels)
    bias = rand(filter_out_channels)
    strides = (1, 2, 2, 1)
    p = ConvolutionLayerParameters(filter, bias, strides)
    @test p.conv2dparams.filter == filter
    @test p.conv2dparams.bias == bias
    @test p.maxpoolparams.strides == strides
    @testset "Base.show" begin
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "convolution layer. applies 5 3x3 filters, followed by max pooling with a 2x2 filter and a stride of (2, 2), and a ReLU activation function."
    end
    @testset "check_size" begin
        @test check_size(p, (3, 3, 1, 5)) === nothing
        @test_throws AssertionError check_size(p, (3, 3, 2, 5))
    end
end

end