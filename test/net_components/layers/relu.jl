using Base.Test
using MIPVerify: ReLU, mip

@testset "relu.jl" begin

@testset "ReLU" begin
    @testset "Initialize without tightening algorithm" begin
        p = ReLU()
        @test isnull(p.tightening_algorithm)
    end

    @testset "Initialize with tightening algorithm" begin
        p = ReLU(mip)
        @test !isnull(p.tightening_algorithm)
        @test get(p.tightening_algorithm) == mip
    end

    @testset "Base.show" begin
        p = ReLU()
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "ReLU()"
    end
end

end