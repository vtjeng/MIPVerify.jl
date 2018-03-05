using Base.Test
using JuMP
using MIPVerify
isdefined(:TestHelpers) || include("../../TestHelpers.jl")
using TestHelpers: get_new_model

@testset "flatten.jl" begin

    @testset "Flatten" begin
        @testset "default permutation" begin
            p = Flatten(5)
            @test p.n_dim == 5
            @test all(p.perm .== [5, 4, 3, 2, 1])
        end
        @testset "specified permutation, legal" begin
            p = Flatten([3, 1, 2, 4])
            @test p.n_dim == 4
            @test all(p.perm .== [3, 1, 2, 4])
        end
        @testset "specified permutation, illegal" begin
            @test_throws DomainError Flatten([3, 1, 4])
        end
        @testset "Base.show" begin
            p = Flatten([3, 1, 2, 4])
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "Flatten(): flattens 4 dimensional input, with dimensions permuted according to the order [3, 1, 2, 4]"
        end
    end

    @testset "flatten" begin
        srand(31415)
        xs = rand(1:5, (2, 2, 2, 2))
        p1 = Flatten([3, 1, 2, 4])
        @test all(p1(xs) .== [4, 3, 3, 1, 4, 5, 3, 5, 2, 4, 5, 5, 4, 2, 5, 4])
        p2 = Flatten([1, 3, 4, 2])
        @test p2(xs) == [4, 3, 3, 1, 2, 5, 4, 5, 4, 3, 5, 5, 4, 5, 2, 4]
    end

end