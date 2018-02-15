using Base.Test
using MIPVerify: FullyConnectedLayerParameters

@testset "fully_connected_layer.jl" begin

@testset "FullyConnectedLayerParameters" begin
    height = 10
    matrix = rand(2, height)
    bias = rand(height)
    p = FullyConnectedLayerParameters(matrix, bias)
    @test p.mmparams.matrix == matrix
    @test p.mmparams.bias == bias
    @testset "Base.show" begin
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "fully connected layer with 2 inputs and 10 output units, and a ReLU activation function."
    end
end

end
