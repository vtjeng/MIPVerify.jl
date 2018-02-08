using MIPVerify: Conv2DParameters, PoolParameters, MaxPoolParameters, AveragePoolParameters, ConvolutionLayerParameters, MatrixMultiplicationParameters, SoftmaxParameters, FullyConnectedLayerParameters
using Base.Test
using Base.Test: @test_throws

using JuMP

@testset "net_parameters.jl" begin

@testset "LayerParameters" begin

@testset "Conv2DParameters" begin
    @testset "With Bias" begin
        @testset "Matched Size" begin
            out_channels = 5
            filter = rand(3, 3, 2, out_channels)
            bias = rand(out_channels)
            p = Conv2DParameters(filter, bias)
            @test p.filter == filter
            @test p.bias == bias
        end
        @testset "Unmatched Size" begin
            filter_out_channels = 4
            bias_out_channels = 5
            filter = rand(3, 3, 2, filter_out_channels)
            bias = rand(bias_out_channels)
            @test_throws AssertionError Conv2DParameters(filter, bias)
        end
    end
    @testset "No Bias" begin
        filter = rand(3, 3, 2, 5)
        p = Conv2DParameters(filter)
        @test p.filter == filter
    end
    @testset "JuMP Variables" begin
        m = Model()
        filter_size = (3, 3, 2, 5)
        filter = map(_ -> @variable(m), CartesianRange(filter_size))
        p = Conv2DParameters(filter)
        @test p.filter == filter
    end
end

@testset "PoolParameters" begin
    strides = (1, 2, 2, 1)
    p = PoolParameters(strides, x -> x)
    @test p.strides == strides
end

@testset "ConvolutionLayerParameters" begin
    filter_out_channels = 5
    filter = rand(3, 3, 1, filter_out_channels)
    bias = rand(filter_out_channels)
    strides = (1, 2, 2, 1)
    p = ConvolutionLayerParameters(filter, bias, strides)
    @test p.conv2dparams.filter == filter
    @test p.conv2dparams.bias == bias
    @test p.maxpoolparams.strides == strides
end

@testset "MatrixMultiplicationParameters" begin
    @testset "With Bias" begin
        @testset "Matched Size" begin
            height = 10
            matrix = rand(2, height)
            bias = rand(height)
            p = MatrixMultiplicationParameters(matrix, bias)
            @test p.matrix == matrix
            @test p.bias == bias
        end
        @testset "Unmatched Size" begin
            matrix_height = 10
            bias_height = 5
            matrix = rand(2, matrix_height)
            bias = rand(bias_height)
            @test_throws AssertionError MatrixMultiplicationParameters(matrix, bias)
        end
    end
end

@testset "SoftmaxParameters" begin
    height = 10
    matrix = rand(2, height)
    bias = rand(height)
    p = SoftmaxParameters(matrix, bias)
    @test p.mmparams.matrix == matrix
    @test p.mmparams.bias == bias
end

@testset "FullyConnectedLayerParameters" begin
    height = 10
    matrix = rand(2, height)
    bias = rand(height)
    p = FullyConnectedLayerParameters(matrix, bias)
    @test p.mmparams.matrix == matrix
    @test p.mmparams.bias == bias
end
end
end 