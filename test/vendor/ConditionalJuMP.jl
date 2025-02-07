using Test

using JuMP
using IntervalArithmetic
using MIPVerify: owner_model, lower_bound, upper_bound
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "ConditionalJuMP.jl" begin
    @testset "owner_model" begin
        m = Model()
        x = @variable(m)
        @test owner_model(x) == m

        y = @variable(m)
        @test owner_model(3 * x + 5 * y) == m
        @test owner_model([x, y]) == m
        @test owner_model([2 * x + 3 * y, 3 * x + 5 * y]) == m
    end

    @testset "interval, lower_bound, upper_bound for JuMP classes" begin
        m = Model()
        @variable(m, 1 <= x <= 3)
        @variable(m, 2 <= y <= 5)
        e = 2 * x + 1

        @testset "JuMP.VariableRef" begin
            @test lower_bound(x) == 1
            @test upper_bound(x) == 3

            i = IntervalArithmetic.interval(x)
            @test lower_bound(i) == 1
            @test upper_bound(i) == 3
        end

        @testset "JuMP.GenericAffExpr" begin
            @test lower_bound(e) == 3
            @test upper_bound(e) == 7

            @test lower_bound(e - 2x) == 1
            @test upper_bound(e - 2x) == 1

            @test lower_bound(2x - 3y) == -13
            @test upper_bound(2x - 3y) == 0

            @test lower_bound(AffExpr(2)) == 2
            @test upper_bound(AffExpr(2)) == 2

            i = IntervalArithmetic.interval(e)
            @test lower_bound(i) == lower_bound(e)
            @test upper_bound(i) == upper_bound(e)
        end
    end

    @testset "interval, lower_bound, upper_bound for regular values" begin
        @testset "Number" begin
            # Test both integer and floating point inputs
            @test lower_bound(5) == 5
            @test upper_bound(5) == 5
            @test lower_bound(3.14) == 3.14
            @test upper_bound(3.14) == 3.14
        end

        @testset "Interval" begin
            i = IntervalArithmetic.interval(5.0)
            @test lower_bound(i) == 5.0
            @test upper_bound(i) == 5.0
        end
    end
end
