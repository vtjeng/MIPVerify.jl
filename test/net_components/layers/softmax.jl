using Base.Test
using MIPVerify: SoftmaxParameters

@testset "softmax.jl" begin

@testset "SoftmaxParameters" begin
    height = 10
    matrix = rand(2, height)
    bias = rand(height)
    p = SoftmaxParameters(matrix, bias)
    @test p.mmparams.matrix == matrix
    @test p.mmparams.bias == bias
    @testset "Base.show" begin
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "softmax layer with 2 inputs and 10 output units."
    end
end

end