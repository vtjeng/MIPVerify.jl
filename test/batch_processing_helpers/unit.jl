using Base.Test
using MIPVerify
using MIPVerify: BatchRunParameters, UnrestrictedPerturbationFamily, mkpath_if_not_present, create_summary_file_if_not_present, verify_target_indices
using MIPVerify: run_on_sample_for_untargeted_attack, run_on_sample_for_targeted_attack
using DataFrames

@testset "unit" begin
    mnist = read_datasets("MNIST")

    @testset "BatchRunParameters" begin
        brp = BatchRunParameters(
            Sequential([], "name"),
            UnrestrictedPerturbationFamily(),
            1,
            0
        )
        @testset "Base.show" begin
            io = IOBuffer()
            Base.show(io, brp)
            @test String(take!(io)) == "name__unrestricted__1__0"
        end
    end

    @testset "mkpath_if_not_present" begin
        mktempdir() do dir
            path = joinpath(dir, "1")
            mkpath_if_not_present(path)
            @test ispath(path)
            mkpath_if_not_present(path)
            @test ispath(path)
        end
    end

    @testset "create_summary_file_if_not_present" begin
        mktempdir() do dir
            file_path = joinpath(dir, "summary.csv")
            create_summary_file_if_not_present(file_path)
            @test isfile(file_path)
        end
    end
    
    @testset "verify_target_indices" begin
        @test_throws AssertionError verify_target_indices([0], mnist.test) 
        @test_throws AssertionError verify_target_indices([10001], mnist.test) 
    end

    @testset "run_on_sample_for_untargeted_attack" begin
        sample_numbers = [101, 102, 103, 104]
        dt = DataFrame(
            SampleNumber = [101, 102, 102, 103, 103],
            SolveStatus = ["Infeasible", "Optimal", "UserObjLimit", "UserObjLimit", "UserLimit"],
            ObjectiveValue = [NaN, 0.8, 0.9, 0.99, NaN])
        expected_results = Dict(
            MIPVerify.never => [false, false, false, true],
            MIPVerify.always => [true, true, true, true],
            MIPVerify.resolve_ambiguous_cases => [false, false, true, true],
            MIPVerify.refine_insecure_cases => [false, true, false, true]
        )
        for solve_rerun_option in keys(expected_results)
            @testset "Solve rerun option: $solve_rerun_option" begin
                actual = map(x -> run_on_sample_for_untargeted_attack(x, dt, solve_rerun_option), sample_numbers)
                expected = expected_results[solve_rerun_option]
                @test actual==expected                
            end
        end
    end

    @testset "run_on_sample_for_targeted_attack" begin
        sample_numbers = [(101, 1), (102, 1), (103, 1), (104, 1), (101, 2)]
        dt = DataFrame(
            SampleNumber = [101, 102, 102, 103, 103, 101],
            TargetIndexes = map(x -> "[$x]", [1, 1, 1, 1, 1, 2]),
            SolveStatus = ["Infeasible", "Optimal", "UserObjLimit", "UserObjLimit", "UserLimit", "Optimal"],
            ObjectiveValue = [NaN, 0.8, 0.9, 0.99, NaN, 0.1])
        expected_results = Dict(
            MIPVerify.never => [false, false, false, true, false],
            MIPVerify.always => [true, true, true, true, true],
            MIPVerify.resolve_ambiguous_cases => [false, false, true, true, false],
            MIPVerify.refine_insecure_cases => [false, true, false, true, false],
            MIPVerify.retarget_infeasible_cases => [true, false, false, true, false]
        )
        for solve_rerun_option in keys(expected_results)
            @testset "Solve rerun option: $solve_rerun_option" begin
                actual = map(x -> run_on_sample_for_targeted_attack(x..., dt, solve_rerun_option), sample_numbers)
                expected = expected_results[solve_rerun_option]
                @test actual==expected                
            end
        end
    end
end