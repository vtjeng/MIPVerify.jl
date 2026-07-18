using Test
using MIPVerify
using MIPVerify: UnrestrictedPerturbationFamily
using HiGHS
using JuMP

@isdefined(TestHelpers) || include("../../TestHelpers.jl")

@testset "margin" begin
    # Simple neural network for testing
    # Input layer: 2 neurons
    # Hidden layer 1: 2 neurons, ReLU activation
    # Output layer: 2 neurons, Linear activation
    # Weights and biases are set to simple values for predictable behavior.
    w1 = [1.0 0.0; 0.0 1.0] # Neuron 1 takes input 1, Neuron 2 takes input 2
    b1 = [0.0, 0.0]
    layer1 = MIPVerify.Linear(w1, b1)
    layer2 = MIPVerify.ReLU()
    w2 = [1.0 -1.0; -1.0 1.0]
    b2 = [0.0, 0.0]
    layer3 = MIPVerify.Linear(w2, b2)
    nn = MIPVerify.Sequential([layer1, layer2, layer3], "SimpleNet")
    input = [1.0, 0.5]
    # Expected output for this input without perturbation:
    # layer1_input = [x1, x2]
    # layer1_output = layer1 * input + b1 = [x1, x2]
    # layer2_output = ReLU(layer1_output) 
    #                 = ReLU([x1, x2]) 
    #                 = [max(0, x1), max(0, x2)]
    # layer3_input = [max(0, x1), max(0, x2)]
    # layer3_output = layer3 * layer2_output + b2 
    #                 = [max(0, x1) - max(0, x2), max(0, x2) - max(0, x1)]

    main_solve_options = Dict() # Default to empty dict, add options if needed separately

    @testset "closest objective" begin
        # For input [1.0, 0.5], original output is [0.5, -0.5].
        # Without perturbation: predicted label is 1, logits margin is -1.

        target_label = 2
        non_target_label = 1

        for (test_name, margin, requires_perturbation) in [
            ("positive margin", 0.5, true),
            ("zero margin", 0.0, true),
            ("negative margin", -0.5, true),
            ("negative margin, no perturbation", -1.5, false),
        ]
            @testset "$test_name" begin
                d = find_adversarial_example(
                    nn,
                    input,
                    target_label,
                    TestHelpers.get_optimizer(),
                    main_solve_options,
                    norm_order = 1,
                    adversarial_example_objective = MIPVerify.closest,
                    margin = margin,
                    pp = UnrestrictedPerturbationFamily(),
                    solve_if_predicted_in_targeted = true,
                )
                @test d[:AdversarialExampleObjective] == MIPVerify.closest
                @test d[:SolveStatus] == MOI.OPTIMAL
                @test d[:WitnessAvailable]
                @test d[:WitnessTargetVerified]
                @test d[:WitnessPerturbationVerified]
                @test d[:WitnessVerified]
                perturbed_input = JuMP.value.(d[:PerturbedInput])
                perturbed_output = perturbed_input |> nn
                @test d[:PerturbedInputValue] == perturbed_input
                @test d[:WitnessOutput] == perturbed_output
                target_logit = perturbed_output[target_label]
                non_target_logit = perturbed_output[non_target_label]
                @test d[:WitnessMargin] ≈ target_logit - non_target_logit
                objective_value = JuMP.objective_value(d[:Model])
                if requires_perturbation
                    @test isapprox(target_logit - non_target_logit, margin; atol = 1e-6)
                    @test !isapprox(perturbed_input, input; atol = 1e-6)
                    @test objective_value > 0
                else
                    @test isapprox(target_logit - non_target_logit, -1; atol = 1e-6)
                    @test isapprox(perturbed_input, input; atol = 1e-6)
                    @test isapprox(objective_value, 0; atol = 1e-6)
                end
            end
        end
    end

    @testset "worst objective" begin
        target_label = 2
        non_target_label = 1

        # Since we are going for the worst objective, the objective value should
        # be the maximum possible difference. Despite the name
        # `UnrestrictedPerturbationFamily`, since each element of the input is
        # constrained to be in [0, 1] (see `get_perturbation_specific_keys`), the
        # maximum difference in the logits is in fact 2.

        for (test_name, margin, is_feasible) in
            [("positive margin, infeasible", 2.5, false), ("positive margin", 1.5, true)]
            @testset "$test_name" begin
                d = find_adversarial_example(
                    nn,
                    input,
                    target_label,
                    TestHelpers.get_optimizer(),
                    main_solve_options,
                    norm_order = 1,
                    adversarial_example_objective = MIPVerify.worst,
                    margin = margin,
                    pp = UnrestrictedPerturbationFamily(),
                    solve_if_predicted_in_targeted = true,
                )
                @test d[:AdversarialExampleObjective] == MIPVerify.worst
                if is_feasible
                    @test d[:SolveStatus] == MOI.OPTIMAL
                    @test d[:WitnessAvailable]
                    @test d[:WitnessTargetVerified]
                    @test d[:WitnessPerturbationVerified]
                    @test d[:WitnessVerified]
                else
                    @test d[:SolveStatus] == MOI.INFEASIBLE
                    @test !d[:WitnessAvailable]
                    @test !d[:WitnessTargetVerified]
                    @test !d[:WitnessPerturbationVerified]
                    @test !d[:WitnessVerified]
                end
                if is_feasible
                    perturbed_input = JuMP.value.(d[:PerturbedInput])
                    perturbed_output = perturbed_input |> nn
                    target_logit = perturbed_output[target_label]
                    non_target_logit = perturbed_output[non_target_label]
                    objective_value = JuMP.objective_value(d[:Model])
                    @test isapprox(objective_value, target_logit - non_target_logit; atol = 1e-6)
                    @test isapprox(objective_value, 2; atol = 1e-6)
                end
            end
        end
    end

    @testset "witness target condition" begin
        # A 0.2 target advantage exercises an ordinary verified witness.
        gap, verified = MIPVerify.witness_satisfies_target([0.4, 0.6], [2], 0.0)
        @test gap ≈ 0.2
        @test verified

        # Equal logits exercise the documented zero-margin tie semantics.
        tie_gap, tie_verified = MIPVerify.witness_satisfies_target([0.5, 0.5], [2], 0.0)
        @test tie_gap == 0.0
        @test tie_verified

        # The target is 0.2 behind, which must not verify as a zero-margin witness.
        failed_gap, failed = MIPVerify.witness_satisfies_target([0.6, 0.4], [2], 0.0)
        @test failed_gap ≈ -0.2
        @test !failed

        # A 1e-15 shortfall exercises only the documented comparison tolerance,
        # far below a solver's usual primal feasibility tolerance.
        _, rounded_boundary_verified =
            MIPVerify.witness_satisfies_target([0.0, 1e-6 - 1e-15], [2], 1e-6)
        @test rounded_boundary_verified

        # Non-finite logits cannot provide a checked classifier output.
        _, nonfinite_verified = MIPVerify.witness_satisfies_target([0.0, NaN], [2], 0.0)
        @test !nonfinite_verified
    end

    @testset "witness component diagnostics" begin
        identity_nn = Sequential([], "witness-component-diagnostics")
        pp = UnrestrictedPerturbationFamily()

        # Candidate [1.1, 0.0] gives target 1 a positive margin but violates the [0, 1] domain.
        perturbation_failure = Dict{Symbol,Any}(:TargetIndexes => [1])
        MIPVerify.record_witness!(
            perturbation_failure,
            identity_nn,
            [0.4, 0.6],
            [1.1, 0.0],
            pp,
            0.0,
        )
        @test perturbation_failure[:WitnessTargetVerified]
        @test !perturbation_failure[:WitnessPerturbationVerified]
        @test !perturbation_failure[:WitnessVerified]

        # Candidate [0.3, 0.7] stays in-domain but leaves target 1 behind target 2 by 0.4.
        target_failure = Dict{Symbol,Any}(:TargetIndexes => [1])
        MIPVerify.record_witness!(target_failure, identity_nn, [0.4, 0.6], [0.3, 0.7], pp, 0.0)
        @test !target_failure[:WitnessTargetVerified]
        @test target_failure[:WitnessPerturbationVerified]
        @test !target_failure[:WitnessVerified]
    end

    @testset "record_witness! ignores stale witness values" begin
        # A 1x1 identity blur kernel reconstructs this unchanged candidate exactly, but the
        # planted zero kernel from an earlier evaluation would fail the channel-sum check if it
        # leaked into the fresh verification.
        blur_input = reshape([0.2, 0.4, 0.6, 0.8], 1, 2, 2, 1)
        stale = Dict{Symbol,Any}(:TargetIndexes => [1], :WitnessBlurKernel => zeros(1, 1, 1, 1))
        MIPVerify.record_witness!(
            stale,
            Sequential([MIPVerify.Flatten(4)], "stale-witness-check"),
            blur_input,
            copy(blur_input),
            MIPVerify.BlurringPerturbationFamily((1, 1)),
            0.0,
        )
        @test stale[:WitnessPerturbationVerified]
        @test stale[:WitnessBlurKernel] == reshape([1.0], 1, 1, 1, 1)
    end

    @testset "margin-aware skipped solve" begin
        # The original input has class-1 margin 1.0. A requested 1.5 margin therefore
        # forces a solve even though class 1 is already the predicted target.
        d = find_adversarial_example(
            nn,
            input,
            1,
            TestHelpers.get_optimizer(),
            main_solve_options,
            norm_order = 1,
            margin = 1.5,
            pp = UnrestrictedPerturbationFamily(),
            solve_if_predicted_in_targeted = false,
        )
        @test haskey(d, :Model)
        @test d[:WitnessVerified]
        @test d[:WitnessMargin] ≈ 1.5 atol = 1e-6

        # A requested 0.5 margin is already met by the original 1.0 gap, so this
        # case exercises the checked zero-distance fast path.
        skipped = find_adversarial_example(
            nn,
            input,
            1,
            TestHelpers.get_optimizer(),
            main_solve_options,
            norm_order = 1,
            margin = 0.5,
            pp = UnrestrictedPerturbationFamily(),
            solve_if_predicted_in_targeted = false,
        )
        @test !haskey(skipped, :Model)
        @test skipped[:AdversarialExampleObjective] == MIPVerify.closest
        @test skipped[:WitnessTargetVerified]
        @test skipped[:WitnessPerturbationVerified]
        @test skipped[:WitnessVerified]
        @test skipped[:WitnessDistance] == 0.0
        @test skipped[:PerturbedInputValue] == input

        # Class 1 is still predicted at x1 = 1.1, but that value is outside the perturbation
        # family's [0, 1] input domain. The fast path must reject it and build a model.
        outside_domain = find_adversarial_example(
            nn,
            [1.1, 0.5],
            1,
            TestHelpers.get_optimizer(),
            main_solve_options,
            norm_order = 1,
            pp = UnrestrictedPerturbationFamily(),
            solve_if_predicted_in_targeted = false,
        )
        @test haskey(outside_domain, :Model)
        @test outside_domain[:WitnessVerified]
        @test outside_domain[:PerturbedInputValue] != [1.1, 0.5]
    end

    @testset "feasibility objective" begin
        # Target class 2 starts one logit below class 1, so a feasible witness must
        # change the input and exercise the solver-backed feasibility path.
        d = find_adversarial_example(
            nn,
            input,
            2,
            TestHelpers.get_optimizer(),
            main_solve_options,
            norm_order = 1,
            adversarial_example_objective = MIPVerify.feasibility,
            pp = UnrestrictedPerturbationFamily(),
            solve_if_predicted_in_targeted = false,
        )
        @test d[:AdversarialExampleObjective] == MIPVerify.feasibility
        @test JuMP.objective_sense(d[:Model]) == MOI.FEASIBILITY_SENSE
        @test d[:PrimalStatus] == MOI.FEASIBLE_POINT
        @test d[:WitnessAvailable]
        @test d[:WitnessTargetVerified]
        @test d[:WitnessPerturbationVerified]
        @test d[:WitnessVerified]
    end

    @testset "feasibility objective with a mixed-integer solve" begin
        # Over the input domain [0, 1], the affine value x - 0.5 crosses zero. Its ReLU therefore
        # requires one binary variable instead of reducing to a fixed phase.
        unstable_relu_nn = Sequential(
            [
                Linear(reshape([1.0], 1, 1), [-0.5]),
                ReLU(),
                # Class 1 has the fixed logit 0.1 and class 2 has the ReLU output. Targeting class
                # 2 forces the ReLU output to at least 0.1, so a feasible witness must activate it.
                Linear([0.0 1.0], [0.1, 0.0]),
            ],
            "feasibility-unstable-relu",
        )
        # At x = 0.25 the ReLU output is zero and class 1 wins, so targeting class 2 exercises a
        # solver-produced witness rather than the already-targeted fast path.
        d = find_adversarial_example(
            unstable_relu_nn,
            [0.25],
            2,
            HiGHS.Optimizer,
            Dict("output_flag" => false),
            norm_order = 1,
            adversarial_example_objective = MIPVerify.feasibility,
            pp = UnrestrictedPerturbationFamily(),
            tightening_algorithm = interval_arithmetic,
            solve_if_predicted_in_targeted = false,
        )

        @test JuMP.num_constraints(d[:Model], JuMP.VariableRef, MOI.ZeroOne) == 1
        @test JuMP.objective_sense(d[:Model]) == MOI.FEASIBILITY_SENSE
        @test d[:WitnessAvailable]
        @test d[:WitnessTargetVerified]
        @test d[:WitnessPerturbationVerified]
        @test d[:WitnessVerified]
    end

end
