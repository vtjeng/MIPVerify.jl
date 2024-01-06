using Test

@testset "zero.jl" begin

    @testset "Zero" begin
        @testset "specified input, zero" begin
            height = 15
            matrix = ones(2, height)
            bias = ones(height)
            p = Zero(matrix, bias)
        end
        @testset "Base.show" begin
            p = Zero()
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "Zero()"
        end
    end

end
