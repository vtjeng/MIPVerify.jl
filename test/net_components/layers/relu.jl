using Test
using MIPVerify: ReLU, mip

@testset "relu.jl" begin

@testset "ReLU" begin
    @testset "Initialize without tightening algorithm" begin
        p = ReLU()
        @test p.tightening_algorithm === nothing
    end

    @testset "Initialize with tightening algorithm" begin
        p = ReLU(mip)
        @test !(p.tightening_algorithm === nothing)
        @test p.tightening_algorithm == mip
    end

    @testset "Base.show" begin
        p = ReLU()
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "ReLU()"
    end
end

end