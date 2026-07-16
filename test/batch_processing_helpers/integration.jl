using Test
using CSV
using DataFrames
using MIPVerify
using MIPVerify: LInfNormBoundedPerturbationFamily
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "integration.jl" begin
    # The weights make class 1's logit `1 - x` and class 2's logit `x`, so the
    # prediction changes at x = 0.5. The -1 and 1 slopes move the logits in
    # opposite directions, while biases 1 and 0 put their crossing at 0.5.
    weights = [-1.0 1.0]
    biases = [1.0, 0.0]
    nn = Sequential(
        [
            Flatten(4), # `get_image` returns one sample with four NHWC dimensions.
            Linear(weights, biases),
        ],
        "batch_processing_helpers.integration.two_class",
    )

    class_boundary = 0.5 # Equal logits are the nearest feasible class-2 prediction.
    perturbation_radius = 0.1 # Large enough to move 0.45, but not 0.1, to the boundary.
    robust_input = 0.1 # Its largest allowed value is 0.2, below the 0.5 boundary.
    attackable_input = 0.45 # It needs a 0.05 perturbation, within the 0.1 bound.
    misclassified_input = 0.75 # The network predicts class 2 although its label is class 1.
    images = reshape(
        [robust_input, attackable_input, misclassified_input],
        3, # One image for each of the robust, attackable, and misclassified cases.
        1, # A one-pixel image has height one.
        1, # A one-pixel image has width one.
        1, # A scalar image has one channel.
    )
    labels = zeros(Int, 3) # Zero is class 1; all three labels exercise attacks toward class 2.
    dataset = MIPVerify.LabelledImageDataset(images, labels)

    robust_sample = 1 # This row must produce an infeasible solve.
    attackable_sample = 2 # This row must produce an optimal nonzero attack.
    misclassified_sample = 3 # This row must skip solving because class 2 is already predicted.
    true_class = 1 # Every zero-based label 0 maps to one-indexed class 1.
    target_class = 2 # Class 2 is the only non-true class in this two-class network.
    objective_tolerance = 1e-5 # Allow small solver-dependent error at the class boundary.
    pp = LInfNormBoundedPerturbationFamily(perturbation_radius)

    function read_batch_output(dir)
        main_path = only(filter(isdir, readdir(dir; join = true)))
        summary = DataFrame(CSV.File(joinpath(main_path, "summary.csv")))
        result_files = readdir(joinpath(main_path, "run_results"))
        return main_path, summary, result_files
    end

    row_for_sample(summary, sample) = only(findall(==(sample), summary.SampleNumber))

    @testset "batch_find_untargeted_attack" begin
        mktempdir() do dir
            MIPVerify.batch_find_untargeted_attack(
                nn,
                dataset,
                [robust_sample, attackable_sample, misclassified_sample],
                TestHelpers.get_optimizer(),
                TestHelpers.get_main_solve_options(),
                solve_rerun_option = MIPVerify.never,
                pp = pp,
                norm_order = Inf,
                tightening_algorithm = interval_arithmetic,
                tightening_options = TestHelpers.get_tightening_options(),
                solve_if_predicted_in_targeted = false,
                save_path = dir,
            )

            main_path, summary, result_files = read_batch_output(dir)
            @test summary.SampleNumber == [robust_sample, attackable_sample, misclassified_sample]
            robust_row = row_for_sample(summary, robust_sample)
            attackable_row = row_for_sample(summary, attackable_sample)
            misclassified_row = row_for_sample(summary, misclassified_sample)
            # Solvers may report infeasible-or-unbounded, which remains unresolved rather than a
            # proof of infeasibility even though this fixture is mathematically bounded.
            @test summary.SolveStatus[robust_row] in ["INFEASIBLE", "INFEASIBLE_OR_UNBOUNDED"]
            @test summary.SolveStatus[[attackable_row, misclassified_row]] == ["OPTIMAL", "OPTIMAL"]
            @test summary.IsInfeasible[robust_row] ==
                  (summary.SolveStatus[robust_row] == "INFEASIBLE")
            @test summary.IsInfeasible[[attackable_row, misclassified_row]] == [false, false]
            @test summary.VerdictOnly == [false, false, false]
            @test summary.WitnessAvailable == [false, true, true]
            @test summary.WitnessVerified == [false, true, true]
            @test isnan(summary.WitnessMargin[robust_row])
            @test all(
                summary.WitnessMargin[[attackable_row, misclassified_row]] .>=
                [-objective_tolerance, 0.0],
            )
            # Saved metadata verifies that the batch API forwarded the requested algorithm.
            @test summary.TighteningApproach[[robust_row, attackable_row]] ==
                  [string(interval_arithmetic), string(interval_arithmetic)]
            @test isnan(summary.ObjectiveValue[robust_row])
            @test summary.ObjectiveValue[attackable_row] ≈ class_boundary - attackable_input atol =
                objective_tolerance
            @test summary.ObjectiveValue[misclassified_row] == 0
            @test !ismissing(summary.ResultRelativePath[robust_row])
            @test !ismissing(summary.ResultRelativePath[attackable_row])
            @test ismissing(summary.ResultRelativePath[misclassified_row])
            @test length(result_files) == 2 # Only the two rows that ran a solve write MAT files.

            robust_result =
                MIPVerify.matread(joinpath(main_path, summary.ResultRelativePath[robust_row]))
            @test haskey(robust_result, "WitnessAvailable")
            @test haskey(robust_result, "WitnessVerified")
            @test !haskey(robust_result, "WitnessOutput")

            attackable_result =
                MIPVerify.matread(joinpath(main_path, summary.ResultRelativePath[attackable_row]))
            # Array-valued witnesses stay in the MAT artifact instead of expanding the CSV row.
            @test all(
                key -> haskey(attackable_result, key),
                [
                    "VerdictOnly",
                    "WitnessAvailable",
                    "WitnessVerified",
                    "WitnessMargin",
                    "WitnessDistance",
                    "WitnessOutput",
                    "PerturbedInputValue",
                ],
            )
        end
    end

    @testset "batch_find_untargeted_attack exact refinement" begin
        mktempdir() do dir
            MIPVerify.batch_find_untargeted_attack(
                nn,
                dataset,
                [attackable_sample],
                TestHelpers.get_optimizer(),
                TestHelpers.get_main_solve_options(),
                solve_rerun_option = MIPVerify.never,
                pp = pp,
                norm_order = Inf,
                tightening_algorithm = interval_arithmetic,
                tightening_options = TestHelpers.get_tightening_options(),
                verdict_only = true,
                save_path = dir,
            )

            main_path, summary, result_files = read_batch_output(dir)
            @test summary.VerdictOnly == [true]
            @test summary.WitnessAvailable == [true]
            @test summary.WitnessVerified == [true]
            # The zero-margin target constraint permits a boundary tie, with solver-scale noise.
            @test summary.WitnessMargin[1] >= -objective_tolerance
            @test length(result_files) == 1

            MIPVerify.batch_find_untargeted_attack(
                nn,
                dataset,
                [attackable_sample],
                TestHelpers.get_optimizer(),
                TestHelpers.get_main_solve_options(),
                solve_rerun_option = MIPVerify.refine_insecure_cases,
                pp = pp,
                norm_order = Inf,
                tightening_algorithm = interval_arithmetic,
                tightening_options = TestHelpers.get_tightening_options(),
                verdict_only = false,
                save_path = dir,
            )

            refined_main_path, refined_summary, refined_result_files = read_batch_output(dir)
            @test refined_main_path == main_path
            @test refined_summary.VerdictOnly == [true, false]
            @test refined_summary.WitnessVerified == [true, true]
            @test refined_summary.ObjectiveValue[2] ≈ class_boundary - attackable_input atol =
                objective_tolerance
            @test length(refined_result_files) == 2
        end
    end

    @testset "batch_find_targeted_attack" begin
        mktempdir() do dir
            MIPVerify.batch_find_targeted_attack(
                nn,
                dataset,
                [attackable_sample, misclassified_sample],
                TestHelpers.get_optimizer(),
                TestHelpers.get_main_solve_options(),
                solve_rerun_option = MIPVerify.never,
                pp = pp,
                norm_order = Inf,
                tightening_algorithm = interval_arithmetic,
                tightening_options = TestHelpers.get_tightening_options(),
                solve_if_predicted_in_targeted = false,
                # Class 1 exercises the true-label early skip; class 2 exercises a solve
                # for sample 2 and an already-predicted-target skip for sample 3.
                target_labels = [true_class, target_class],
                save_path = dir,
            )

            _, summary, result_files = read_batch_output(dir)
            @test summary.SampleNumber == [attackable_sample, misclassified_sample]
            @test summary.TargetIndexes == ["[$target_class]", "[$target_class]"]
            attackable_row = row_for_sample(summary, attackable_sample)
            misclassified_row = row_for_sample(summary, misclassified_sample)
            # The attackable row solves; the already-target row takes the zero-objective skip.
            @test summary.SolveStatus == ["OPTIMAL", "OPTIMAL"]
            @test summary.VerdictOnly == [false, false]
            @test summary.WitnessAvailable == [true, true]
            @test summary.WitnessVerified == [true, true]
            @test summary.TighteningApproach[attackable_row] == string(interval_arithmetic)
            @test summary.ObjectiveValue[attackable_row] ≈ class_boundary - attackable_input atol =
                objective_tolerance
            @test summary.ObjectiveValue[misclassified_row] == 0
            @test !ismissing(summary.ResultRelativePath[attackable_row])
            @test ismissing(summary.ResultRelativePath[misclassified_row])
            @test length(result_files) == 1 # Only the targeted solve writes a MAT result.
        end
    end
end
