using Test
using MIPVerify
using MIPVerify:
    BatchRunParameters,
    UnrestrictedPerturbationFamily,
    mkpath_if_not_present,
    create_summary_file_if_not_present,
    is_infeasible,
    read_summary_file,
    verify_target_indices,
    extract_results_for_save
using MIPVerify: run_on_sample_for_untargeted_attack, run_on_sample_for_targeted_attack
using CSV
using DataFrames
using MathOptInterface
using JuMP
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "unit.jl" begin
    # Three one-pixel samples exercise index validation without loading the full MNIST set.
    dataset = MIPVerify.LabelledImageDataset(zeros(3, 1, 1, 1), zeros(Int, 3))

    @testset "BatchRunParameters" begin
        brp = BatchRunParameters(Sequential([], "name"), UnrestrictedPerturbationFamily(), 1)
        @testset "Base.show" begin
            io = IOBuffer()
            Base.show(io, brp)
            @test String(take!(io)) == "name__unrestricted__1"
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
            summary = DataFrame(CSV.File(file_path))
            @test all(
                column -> column in propertynames(summary),
                [:VerdictOnly, :WitnessAvailable, :WitnessVerified, :WitnessMargin],
            )
        end
    end

    @testset "read_summary_file upgrades legacy summaries" begin
        mktempdir() do dir
            file_path = joinpath(dir, "summary.csv")
            # Objective 0.25 represents a legacy incumbent whose witness was not verified. The
            # second row's combined status was historically marked infeasible, but is unresolved.
            CSV.write(
                file_path,
                DataFrame(
                    SampleNumber = [7, 8],
                    SolveStatus = ["TIME_LIMIT", "INFEASIBLE_OR_UNBOUNDED"],
                    IsInfeasible = [false, true],
                    ObjectiveValue = [0.25, NaN],
                ),
            )
            summary = read_summary_file(file_path)
            @test summary.IsInfeasible == [false, false]
            @test summary.VerdictOnly == [false, false]
            @test ismissing(summary.WitnessAvailable[1])
            @test ismissing(summary.WitnessVerified[1])
            @test ismissing(summary.WitnessMargin[1])

            persisted = DataFrame(CSV.File(file_path))
            @test :WitnessVerified in propertynames(persisted)
            @test persisted.IsInfeasible == [false, false]
        end
    end

    @testset "verify_target_indices" begin
        # Zero is below the one-based range; four is one past this three-sample fixture.
        @test_throws AssertionError verify_target_indices([0], dataset)
        @test_throws AssertionError verify_target_indices([4], dataset)
    end

    @testset "is_infeasible" begin
        @test is_infeasible(MathOptInterface.INFEASIBLE)
        # This combined status does not identify which side holds, so it cannot certify robustness.
        @test !is_infeasible(MathOptInterface.INFEASIBLE_OR_UNBOUNDED)
        @test !is_infeasible(MathOptInterface.OPTIMAL)
    end

    @testset "exact refinement rejects verdict-only mode" begin
        sample_index = firstindex(dataset.labels) # Select one valid fixture sample without solving.
        target_label = 1 # Any valid label reaches the mode check before targeted work begins.
        nn = Sequential([], "batch-refinement-mode-check")
        optimizer = TestHelpers.get_optimizer()
        main_solve_options = TestHelpers.get_main_solve_options()

        @test_throws ArgumentError MIPVerify.batch_find_untargeted_attack(
            nn,
            dataset,
            [sample_index],
            optimizer,
            main_solve_options;
            solve_rerun_option = MIPVerify.refine_insecure_cases,
            verdict_only = true,
        )
        @test_throws ArgumentError MIPVerify.batch_find_targeted_attack(
            nn,
            dataset,
            [sample_index],
            optimizer,
            main_solve_options;
            solve_rerun_option = MIPVerify.refine_insecure_cases,
            target_labels = [target_label],
            verdict_only = true,
        )
    end

    @testset "run_on_sample_for_untargeted_attack" begin
        # Samples 101-103 cover proven infeasibility, a legacy incumbent stopped at an
        # objective limit, and a time limit without an incumbent; 104 has no prior row.
        sample_numbers = [101, 102, 103, 104]
        dt = DataFrame(
            SampleNumber = [101, 102, 102, 103, 103],
            SolveStatus = [
                "INFEASIBLE",
                "OPTIMAL",
                "OBJECTIVE_LIMIT",
                "OBJECTIVE_LIMIT",
                "TIME_LIMIT",
            ],
            ObjectiveValue = [NaN, 0.8, 0.9, 0.99, NaN],
        )
        expected_results = Dict(
            MIPVerify.never => [false, false, false, true],
            MIPVerify.always => [true, true, true, true],
            MIPVerify.resolve_ambiguous_cases => [false, false, true, true],
            MIPVerify.refine_insecure_cases => [false, true, false, true],
        )
        for solve_rerun_option in keys(expected_results)
            @testset "Solve rerun option: $solve_rerun_option" begin
                actual = map(
                    x -> run_on_sample_for_untargeted_attack(x, dt, solve_rerun_option),
                    sample_numbers,
                )
                expected = expected_results[solve_rerun_option]
                @test actual == expected
            end
        end
    end

    @testset "verified witness rerun semantics" begin
        # Each row isolates one semantic result: ambiguous infeasible-or-unbounded status, verified
        # witness at a solution limit, rejected witness at an objective limit, unresolved time
        # limit, optimal exact witness, and optimal verdict-only witness. Objective 0.9 on the
        # rejected row checks that explicit witness verification overrides the legacy fallback.
        sample_numbers = [201, 202, 203, 204, 205, 206]
        dt = DataFrame(
            SampleNumber = sample_numbers,
            SolveStatus = [
                "INFEASIBLE_OR_UNBOUNDED",
                "SOLUTION_LIMIT",
                "OBJECTIVE_LIMIT",
                "TIME_LIMIT",
                "OPTIMAL",
                "OPTIMAL",
            ],
            ObjectiveValue = [NaN, 0.8, 0.9, NaN, 0.7, 0.0],
            WitnessAvailable = [false, true, true, false, true, true],
            WitnessVerified = [false, true, false, false, true, true],
            VerdictOnly = [false, false, false, false, false, true],
        )

        resolve_results = map(
            sample -> run_on_sample_for_untargeted_attack(
                sample,
                dt,
                MIPVerify.resolve_ambiguous_cases,
            ),
            sample_numbers,
        )
        # `INFEASIBLE_OR_UNBOUNDED` is not a proof of infeasibility, so it remains ambiguous.
        @test resolve_results == [true, false, true, true, false, false]

        refine_results = map(
            sample -> run_on_sample_for_untargeted_attack(
                sample,
                dt,
                MIPVerify.refine_insecure_cases,
            ),
            sample_numbers,
        )
        @test refine_results == [false, true, false, false, false, true]
    end

    @testset "run_on_sample_for_targeted_attack" begin
        sample_numbers = [(101, 1), (102, 1), (103, 1), (104, 1), (101, 2)]
        dt = DataFrame(
            SampleNumber = [101, 102, 102, 103, 103, 101],
            TargetIndexes = map(x -> "[$x]", [1, 1, 1, 1, 1, 2]),
            SolveStatus = [
                "INFEASIBLE",
                "OPTIMAL",
                "OBJECTIVE_LIMIT",
                "OBJECTIVE_LIMIT",
                "TIME_LIMIT",
                "OPTIMAL",
            ],
            ObjectiveValue = [NaN, 0.8, 0.9, 0.99, NaN, 0.1],
        )
        expected_results = Dict(
            MIPVerify.never => [false, false, false, true, false],
            MIPVerify.always => [true, true, true, true, true],
            MIPVerify.resolve_ambiguous_cases => [false, false, true, true, false],
            MIPVerify.refine_insecure_cases => [false, true, false, true, false],
            MIPVerify.retarget_infeasible_cases => [true, false, false, true, false],
        )
        for solve_rerun_option in keys(expected_results)
            @testset "Solve rerun option: $solve_rerun_option" begin
                actual = map(
                    x -> run_on_sample_for_targeted_attack(x..., dt, solve_rerun_option),
                    sample_numbers,
                )
                expected = expected_results[solve_rerun_option]
                @test actual == expected
            end
        end
    end

    @testset "extract_results_for_save handles no-primal-solution status" begin
        m = TestHelpers.get_new_model()
        x = @variable(m, lower_bound = 0)
        @objective(m, Min, x)
        d = Dict(
            :Model => m,
            :SolveTime => 0.0,
            :SolveStatus => MathOptInterface.TIME_LIMIT,
            :VerdictOnly => true,
            :WitnessAvailable => false,
            :WitnessVerified => false,
            :Perturbation => [x],
            :PerturbedInput => [x],
            :TargetIndexes => [1],
            :PredictedIndex => 1,
            :TighteningApproach => :interval_arithmetic,
            :TotalTime => 0.0,
        )
        result = extract_results_for_save(d)
        @test isnan(result[:ObjectiveValue])
        @test haskey(result, :ObjectiveBound)
        @test !haskey(result, :PerturbationValue)
        @test !haskey(result, :PerturbedInputValue)
        @test result[:VerdictOnly]
        @test !result[:WitnessAvailable]
        @test !result[:WitnessVerified]
        @test !haskey(result, :WitnessMargin)
        @test !haskey(result, :WitnessOutput)
    end

    @testset "extract_results_for_save persists a verified witness" begin
        m = TestHelpers.get_new_model()
        x = @variable(m, lower_bound = 0)
        @objective(m, Min, x)
        optimize!(m)

        # Margin 0.25 and the two logits are distinct sentinels that verify each witness field is
        # copied from the core result instead of reconstructed from the JuMP expressions.
        d = Dict(
            :Model => m,
            :SolveTime => 0.0,
            :SolveStatus => JuMP.termination_status(m),
            :VerdictOnly => false,
            :WitnessAvailable => true,
            :WitnessVerified => true,
            :WitnessMargin => 0.25,
            :WitnessOutput => [0.75, 0.5],
            :WitnessDistance => 0.125,
            :PerturbedInputValue => [0.0],
            :Perturbation => [x],
            :PerturbedInput => [x],
            :TargetIndexes => [1],
            :PredictedIndex => 1,
            :TighteningApproach => :interval_arithmetic,
            :TotalTime => 0.0,
        )
        result = extract_results_for_save(d)
        @test result[:PerturbedInputValue] == [0.0]
        @test result[:PerturbationValue] == [0.0]
        @test result[:WitnessOutput] == [0.75, 0.5]
        @test result[:WitnessMargin] == 0.25
        @test result[:WitnessDistance] == 0.125
        @test result[:WitnessAvailable]
        @test result[:WitnessVerified]
    end
end
