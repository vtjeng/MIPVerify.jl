using Test

include(joinpath(@__DIR__, "..", "BenchmarkHelpers.jl"))
using .BenchmarkHelpers

@testset "BenchmarkHelpers" begin
    @testset "parse_args" begin
        @test parse_args(String[]) == Dict{String,String}()
        @test parse_args(["--out", "/tmp"]) == Dict("out" => "/tmp")
        @test parse_args(["--samples", "1:10", "--out", "/tmp"]) ==
              Dict("samples" => "1:10", "out" => "/tmp")
        @test parse_args(["--verbose"]) == Dict("verbose" => "true")
        @test parse_args(["--verbose", "--out", "/tmp"]) ==
              Dict("verbose" => "true", "out" => "/tmp")
        @test parse_args(["positional", "--key", "val"]) == Dict("key" => "val")
    end

    @testset "parse_sample_spec" begin
        @test parse_sample_spec("1:5") == [1, 2, 3, 4, 5]
        @test parse_sample_spec("1:2:10") == [1, 3, 5, 7, 9]
        @test parse_sample_spec("3,7,11") == [3, 7, 11]
        @test parse_sample_spec("42") == [42]
        @test parse_sample_spec("1:1") == [1]
        @test_throws AssertionError parse_sample_spec("1:2:3:4")
    end

    @testset "safe_sum" begin
        @test safe_sum([1.0, 2.0, 3.0]) == 6.0
        @test safe_sum([1.0, Inf, 3.0]) == 4.0
        @test safe_sum([Inf, -Inf, NaN]) == 0.0
        @test safe_sum([1.0]) == 1.0
    end

    @testset "is_infeasible_status" begin
        @test is_infeasible_status("INFEASIBLE") == true
        @test is_infeasible_status("INFEASIBLE_OR_UNBOUNDED") == true
        @test is_infeasible_status("OPTIMAL") == false
        @test is_infeasible_status("TIME_LIMIT") == false
    end

    @testset "classify_semantic_outcome" begin
        @test classify_semantic_outcome("INFEASIBLE", missing) == "certified_no_adversarial_example"
        @test classify_semantic_outcome("INFEASIBLE_OR_UNBOUNDED", missing) ==
              "certified_no_adversarial_example"
        @test classify_semantic_outcome("OPTIMAL", 0.5) == "adversarial_example_found_or_best_known"
        @test classify_semantic_outcome("INFEASIBLE", 0.5) == "certified_no_adversarial_example"
        @test classify_semantic_outcome("TIME_LIMIT", missing) == "time_limit_unresolved"
        @test classify_semantic_outcome("OTHER", missing) == "no_primal_solution_other"
    end

    @testset "maybe_parse_norm_order" begin
        @test maybe_parse_norm_order("Inf") == Inf
        @test maybe_parse_norm_order("inf") == Inf
        @test maybe_parse_norm_order(" Inf ") == Inf
        @test maybe_parse_norm_order("1") == 1.0
        @test maybe_parse_norm_order("2.0") == 2.0
    end

    @testset "regression_ratio" begin
        @test regression_ratio(100.0, 105.0) == 0.05
        @test regression_ratio(100.0, 100.0) == 0.0
        @test regression_ratio(100.0, 95.0) == -0.05
        @test regression_ratio(0.0, 0.0) == 0.0
        @test regression_ratio(0.0, 1.0) == Inf
    end

    @testset "percent" begin
        @test percent(0.05) == "5.0%"
        @test percent(0.0) == "0.0%"
        @test percent(-0.1) == "-10.0%"
        @test percent(Inf) == "Inf%"
    end
end
