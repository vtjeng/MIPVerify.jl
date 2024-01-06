using Test

@testset "zero.jl" begin

    @testset "Zero" begin
        @testset "specified input, zero" begin
            p = Zero()
            @test p(ones(1)) == 0
        end
        @testset "Base.show" begin
            p = Zero()
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "Zero()"
        end
    end

end
