using Base.Test
using MIPVerify: ReLU

@testset "relu.jl" begin

@testset "ReLU" begin
    @testset "Base.show" begin
        p = ReLU()
        io = IOBuffer()
        Base.show(io, p)
        @test String(take!(io)) == "ReLU()"
    end
end

end