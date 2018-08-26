using Base.Test
using MIPVerify: UnrestrictedPerturbationFamily, BlurringPerturbationFamily, LInfNormBoundedPerturbationFamily

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
    @testset "LInfNormBoundedPerturbationFamily" begin
        @testset "Base.show" begin
            p = LInfNormBoundedPerturbationFamily(0.1)
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "linf-norm-bounded-0.1"
        end
    end
end