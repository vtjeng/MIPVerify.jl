using Test
using MIPVerify: MaskedReLU, mip
using Random

@testset "masked_relu.jl" begin

@testset "MaskedReLU" begin
    mask = rand(Random.MersenneTwister(0), [-1, 0, 1], 10)
    @testset "Initialize without tightening algorithm" begin
        p = MaskedReLU(mask)
        @test p.tightening_algorithm === nothing
    end

    @testset "Initialize with tightening algorithm" begin
        p = MaskedReLU(mask, mip)
        @test !(p.tightening_algorithm === nothing)
        @test p.tightening_algorithm == mip
    end

    @testset "Base.show" begin
        p = MaskedReLU(mask)
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "MaskedReLU with expected input size (10,). (4 zeroed, 3 as-is, 3 rectified)."
    end
end

end