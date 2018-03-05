using Base.Test
using MIPVerify: MaskedReLU

@testset "masked_relu.jl" begin

@testset "MaskedReLU" begin
    @testset "Base.show" begin
        mask = rand(MersenneTwister(0), [-1, 0, 1], 10)
        p = MaskedReLU(mask)
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "MaskedReLU with expected input size (10,). (4 zeroed, 4 as-is, 2 rectified)."
    end
end

end