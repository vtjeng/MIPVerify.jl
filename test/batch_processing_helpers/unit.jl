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
using DelimitedFiles
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
                [
                    :AdversarialExampleObjective,
                    :WitnessAvailable,
                    :WitnessTargetVerified,
                    :WitnessPerturbationVerified,
                    :WitnessVerified,
                    :WitnessMargin,
                ],
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
            @test all(ismissing, summary.AdversarialExampleObjective)
            @test ismissing(summary.WitnessAvailable[1])
            @test ismissing(summary.WitnessTargetVerified[1])
            @test ismissing(summary.WitnessPerturbationVerified[1])
            @test ismissing(summary.WitnessVerified[1])
            @test ismissing(summary.WitnessMargin[1])

            persisted = DataFrame(CSV.File(file_path))
            @test :WitnessVerified in propertynames(persisted)
            @test persisted.IsInfeasible == [false, false]
        end
    end

    @testset "read_summary_file rejects unknown objectives" begin
        mktempdir() do dir
            file_path = joinpath(dir, "summary.csv")
            CSV.write(
                file_path,
                DataFrame(
                    SampleNumber = [1],
                    SolveStatus = ["OPTIMAL"],
                    IsInfeasible = [false],
                    AdversarialExampleObjective = ["unknown"],
                ),
            )
            @test_throws ArgumentError read_summary_file(file_path)
        end
    end

    @testset "read_summary_file rejects the unreleased boolean-objective schema" begin
        mktempdir() do dir
            file_path = joinpath(dir, "summary.csv")
            CSV.write(
                file_path,
                DataFrame(
                    SampleNumber = [1],
                    SolveStatus = ["OPTIMAL"],
                    IsInfeasible = [false],
                    VerdictOnly = [true],
                ),
            )
            @test_throws ArgumentError read_summary_file(file_path)
        end
    end

    @testset "resuming a migrated summary preserves witness columns" begin
        mktempdir() do dir
            file_path = joinpath(dir, "summary.csv")
            # This row uses the complete target-only witness schema that preceded the two
            # component columns. Distinct Boolean values expose any positional column shift.
            legacy_summary = DataFrame(
                SampleNumber = [1],
                ResultRelativePath = ["run_results/legacy.mat"],
                PredictedIndex = [1],
                TargetIndexes = ["[2]"],
                SolveTime = [0.5],
                SolveStatus = ["OPTIMAL"],
                IsInfeasible = [false],
                ObjectiveValue = [0.25],
                ObjectiveBound = [0.2],
                TighteningApproach = ["lp"],
                TotalTime = [1.0],
                WitnessAvailable = [true],
                WitnessVerified = [true],
                WitnessMargin = [0.1],
            )
            CSV.write(file_path, legacy_summary)

            migrated = read_summary_file(file_path)
            @test propertynames(migrated) == Symbol.(MIPVerify.SUMMARY_HEADER)
            @test ismissing(migrated.AdversarialExampleObjective[1])
            @test ismissing(migrated.WitnessTargetVerified[1])
            @test ismissing(migrated.WitnessPerturbationVerified[1])

            # The second row's 0.3 margin and false target check are sentinels for the new
            # positional layout when a resumed batch appends to the migrated file.
            resumed_result = Dict(
                :PredictedIndex => 2,
                :TargetIndexes => [1],
                :SolveTime => 0.75,
                :SolveStatus => MathOptInterface.OPTIMAL,
                :ObjectiveValue => 0.4,
                :ObjectiveBound => 0.35,
                :TighteningApproach => "mip",
                :TotalTime => 1.25,
                :AdversarialExampleObjective => "closest",
                :WitnessAvailable => true,
                :WitnessTargetVerified => false,
                :WitnessPerturbationVerified => true,
                :WitnessVerified => false,
                :WitnessMargin => 0.3,
            )
            summary_line =
                MIPVerify.generate_csv_summary_line(2, "run_results/resumed.mat", resumed_result)
            open(file_path, "a") do file
                writedlm(file, [summary_line], ',')
            end

            resumed = DataFrame(CSV.File(file_path))
            @test ismissing(resumed.AdversarialExampleObjective[1])
            @test resumed.AdversarialExampleObjective[2] == "closest"
            @test resumed.WitnessAvailable[2]
            @test !resumed.WitnessTargetVerified[2]
            @test resumed.WitnessPerturbationVerified[2]
            @test !resumed.WitnessVerified[2]
            @test resumed.WitnessMargin[2] == 0.3
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

    @testset "exact refinement rejects the feasibility objective" begin
        sample_index = firstindex(dataset.labels) # Select one valid fixture sample without solving.
        target_label = 1 # Any valid label reaches the objective check before targeted work begins.
        nn = Sequential([], "batch-refinement-objective-check")
        optimizer = TestHelpers.get_optimizer()
        main_solve_options = TestHelpers.get_main_solve_options()

        @test_throws ArgumentError MIPVerify.batch_find_untargeted_attack(
            nn,
            dataset,
            [sample_index],
            optimizer,
            main_solve_options;
            solve_rerun_option = MIPVerify.refine_insecure_cases,
            adversarial_example_objective = MIPVerify.feasibility,
        )
        @test_throws ArgumentError MIPVerify.batch_find_targeted_attack(
            nn,
            dataset,
            [sample_index],
            optimizer,
            main_solve_options;
            solve_rerun_option = MIPVerify.refine_insecure_cases,
            target_labels = [target_label],
            adversarial_example_objective = MIPVerify.feasibility,
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
            MIPVerify.resolve_ambiguous_cases => [false, true, true, true],
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
        # limit, optimal exact witness, optimal feasibility witness, and a rejected witness paired
        # with a contradictory infeasible status. Objective 0.9 on the objective-limit row checks
        # that an explicit perturbation-check failure overrides legacy objective evidence.
        sample_numbers = [201, 202, 203, 204, 205, 206, 207]
        dt = DataFrame(
            SampleNumber = sample_numbers,
            SolveStatus = [
                "INFEASIBLE_OR_UNBOUNDED",
                "SOLUTION_LIMIT",
                "OBJECTIVE_LIMIT",
                "TIME_LIMIT",
                "OPTIMAL",
                "OPTIMAL",
                "INFEASIBLE",
            ],
            ObjectiveValue = [NaN, 0.8, 0.9, NaN, 0.7, 0.0, NaN],
            WitnessAvailable = [false, true, true, false, true, true, true],
            WitnessTargetVerified = [false, true, true, false, true, true, false],
            WitnessPerturbationVerified = [false, true, false, false, true, true, true],
            WitnessVerified = [false, true, false, false, true, true, false],
            AdversarialExampleObjective = Union{Missing,String}[
                missing,
                missing,
                missing,
                missing,
                "closest",
                "feasibility",
                missing,
            ],
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
        @test resolve_results == [true, false, true, true, false, false, true]

        refine_results = map(
            sample -> run_on_sample_for_untargeted_attack(
                sample,
                dt,
                MIPVerify.refine_insecure_cases,
            ),
            sample_numbers,
        )
        @test refine_results == [false, true, false, false, false, true, false]
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
            MIPVerify.resolve_ambiguous_cases => [false, true, true, true, true],
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
            :AdversarialExampleObjective => MIPVerify.feasibility,
            :WitnessAvailable => false,
            :WitnessTargetVerified => false,
            :WitnessPerturbationVerified => false,
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
        @test result[:AdversarialExampleObjective] == "feasibility"
        @test !result[:WitnessAvailable]
        @test !result[:WitnessTargetVerified]
        @test !result[:WitnessPerturbationVerified]
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
            :AdversarialExampleObjective => MIPVerify.closest,
            :WitnessAvailable => true,
            :WitnessTargetVerified => true,
            :WitnessPerturbationVerified => true,
            :WitnessVerified => true,
            :WitnessMargin => 0.25,
            :WitnessOutput => [0.75, 0.5],
            :WitnessDistance => 0.125,
            # A one-channel 1x1 identity kernel is a compact sentinel for blur auxiliary data.
            :WitnessBlurKernel => reshape([1.0], 1, 1, 1, 1),
            :PerturbedInputValue => [0.0],
            :Perturbation => [x],
            :PerturbedInput => [x],
            :TargetIndexes => [1],
            :PredictedIndex => 1,
            :TighteningApproach => :interval_arithmetic,
            :TotalTime => 0.0,
        )
        result = extract_results_for_save(d)
        @test result[:AdversarialExampleObjective] == "closest"
        @test result[:PerturbedInputValue] == [0.0]
        @test result[:PerturbationValue] == [0.0]
        @test result[:WitnessOutput] == [0.75, 0.5]
        @test result[:WitnessMargin] == 0.25
        @test result[:WitnessDistance] == 0.125
        @test result[:WitnessBlurKernel] == reshape([1.0], 1, 1, 1, 1)
        @test result[:WitnessAvailable]
        @test result[:WitnessTargetVerified]
        @test result[:WitnessPerturbationVerified]
        @test result[:WitnessVerified]
    end
end
