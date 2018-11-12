using Base.Test
using MIPVerify: MaskedReLU, mip, DEFAULT_TIGHTENING_ALGORITHM_SEQUENCE

@testset "masked_relu.jl" begin

@testset "MaskedReLU" begin
    mask = rand(MersenneTwister(0), [-1, 0, 1], 10)
    @testset "Initialize without tightening algorithm" begin
        p = MaskedReLU(mask)
        @test p.tightening_algorithms == DEFAULT_TIGHTENING_ALGORITHM_SEQUENCE
    end

    @testset "Initialize with single tightening algorithm" begin
        p = MaskedReLU(mask, mip)
        @test p.tightening_algorithms == [mip]
    end

    @testset "Initialize with tightening algorithm sequence" begin
        p = MaskedReLU(mask,[mip])
        @test p.tightening_algorithms == [mip]
    end

    @testset "Base.show" begin
        p = MaskedReLU(mask)
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "MaskedReLU with expected input size (10,). (4 zeroed, 4 as-is, 2 rectified)."
    end
end

end