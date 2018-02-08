using Base.Test
using MIPVerify: AdditivePerturbationParameters, BlurPerturbationParameters

@testset "models.jl" begin
    @testset "AdditivePerturbationParameters" begin
        @testset "Base.show" begin
            p = AdditivePerturbationParameters()
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "additive"
        end
    end
    @testset "BlurPerturbationParameters" begin
        @testset "Base.show" begin
            p = BlurPerturbationParameters((5,5))
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "blur.(5, 5)"
        end
    end
end