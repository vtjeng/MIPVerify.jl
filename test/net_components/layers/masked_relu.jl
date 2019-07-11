using Test
using MIPVerify: MaskedReLU, mip

@testset "masked_relu.jl" begin

@testset "MaskedReLU" begin
    mask = [-1, 1, 0, 0, 1, 1]
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
        @test String(take!(io)) == "MaskedReLU with expected input size (6,). (1 zeroed, 3 as-is, 2 rectified)."
    end
end

end