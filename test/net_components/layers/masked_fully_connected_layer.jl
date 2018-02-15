using Base.Test
using MIPVerify: MaskedFullyConnectedLayerParameters

@testset "masked_fully_connected_layer.jl" begin

@testset "MaskedFullyConnectedLayerParameters" begin
    height = 10
    matrix = rand(2, height)
    bias = rand(height)
    mask = [-1, 0, 0, 0, 0, -1, -1, 1, 1, 0]
    p = MaskedFullyConnectedLayerParameters(matrix, bias, mask)
    @test p.mmparams.matrix == matrix
    @test p.mmparams.bias == bias
    @test p.mask == mask
    @testset "Base.show" begin
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "masked fully connected layer with 2 inputs and 10 output units (3 zeroed, 2 as-is, 5 rectified)."
    end
end

end