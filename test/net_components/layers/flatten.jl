using Test
using MIPVerify

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
            @test String(take!(io)) ==
                  "Flatten(): flattens 4 dimensional input, with dimensions permuted according to the order [3, 1, 2, 4]"
        end
    end

    @testset "Flatten" begin
        xs = reshape(collect(1:16), (2, 2, 2, 2))
        p1 = Flatten([3, 1, 2, 4])
        @test p1(xs) == [1, 5, 2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16]
        p2 = Flatten([1, 3, 4, 2])
        @test p2(xs) == [1, 2, 5, 6, 9, 10, 13, 14, 3, 4, 7, 8, 11, 12, 15, 16]
        p3 = Flatten([1, 2, 3, 4])
        @test p3(xs) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    end

end
