using Base.Test
using MIPVerify: ReLU, mip, DEFAULT_TIGHTENING_ALGORITHM_SEQUENCE

@testset "relu.jl" begin

@testset "ReLU" begin
    @testset "Initialize without tightening algorithm" begin
        p = ReLU()
        @test p.tightening_algorithms == DEFAULT_TIGHTENING_ALGORITHM_SEQUENCE
    end

    @testset "Initialize with single tightening algorithm" begin
        p = ReLU(mip)
        @test p.tightening_algorithms == (mip,)
    end

    @testset "Initialize with tightening algorithm" begin
        p = ReLU((mip, ))
        @test p.tightening_algorithms == (mip,)
    end

    @testset "Base.show" begin
        p = ReLU()
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "ReLU()"
    end
end

end