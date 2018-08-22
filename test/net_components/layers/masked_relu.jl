using Base.Test
using MIPVerify: MaskedReLU, mip

@testset "masked_relu.jl" begin

@testset "MaskedReLU" begin
    mask = rand(MersenneTwister(0), [-1, 0, 1], 10)
    @testset "Initialize without tightening algorithm" begin
        p = MaskedReLU(mask)
        @test isnull(p.tightening_algorithm)
    end

    @testset "Initialize with tightening algorithm" begin
        p = MaskedReLU(mask, mip)
        @test !isnull(p.tightening_algorithm)
        @test get(p.tightening_algorithm) == mip
    end

    @testset "Base.show" begin
        p = MaskedReLU(mask)
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "MaskedReLU with expected input size (10,). (4 zeroed, 4 as-is, 2 rectified)."
    end
end

end