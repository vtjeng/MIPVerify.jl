using Base.Test
using MIPVerify: setloglevel!
using MIPVerify: remove_cached_models
using MIPVerify: get_max_index, get_norm
using JuMP

isdefined(:TestHelpers) || include("TestHelpers.jl")
using TestHelpers: get_new_model

@testset "MIPVerify" begin
    MIPVerify.setloglevel!("info")
    MIPVerify.remove_cached_models()

    include("integration.jl")
    include("net_components.jl")
    include("utils.jl")
    include("models.jl")
    include("batch_processing_helpers.jl")
    
    @testset "get_max_index" begin
        @test_throws MethodError get_max_index([])
        @test get_max_index([3]) == 1
        @test get_max_index([3, 1, 4]) == 3
        @test get_max_index([3, 1, 4, 1, 5, 9, 2]) == 6
    end

    @testset "get_norm" begin
        @testset "real-valued arrays" begin
            xs = [1, -2, 3]
            @test get_norm(1, xs) == 6
            @test get_norm(2, xs) == sqrt(14)
            @test get_norm(Inf, xs) == 3
            @test_throws DomainError get_norm(3, xs)
        end
        @testset "variable-valued arrays" begin
            @testset "l1" begin
                m = get_new_model()
                x1 = @variable(m, lowerbound=1, upperbound=5)
                x2 = @variable(m, lowerbound=-8, upperbound=-2)
                x3 = @variable(m, lowerbound=3, upperbound=10)
                xs = [x1, x2, x3]
                n_1 = get_norm(1, xs)
                n_2 = get_norm(2, xs)
                n_inf = get_norm(Inf, xs)

                @objective(m, Min, n_1)
                solve(m)
                @test getobjectivevalue(m)≈6

                if Pkg.installed("Gurobi") != nothing
                    # Skip these tests if Gurobi is not installed.
                    # Cbc does not solve problems with quadratic objectives
                    @objective(m, Min, n_2)
                    solve(m)
                    @test getobjectivevalue(m)≈14
                end

                @objective(m, Min, n_inf)
                solve(m)
                @test getobjectivevalue(m)≈3

                @test_throws DomainError get_norm(3, xs)
            end
        end
    end
end

