using Base.Test
using MIPVerify: UnrestrictedPerturbationFamily, BlurringPerturbationFamily

@testset "models.jl" begin
    @testset "UnrestrictedPerturbationFamily" begin
        @testset "Base.show" begin
            p = UnrestrictedPerturbationFamily()
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "unrestricted"
        end
    end
    @testset "BlurringPerturbationFamily" begin
        @testset "Base.show" begin
            p = BlurringPerturbationFamily((5,5))
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "blur-(5,5)"
        end
    end
end