using Test
using MIPVerify
using MIPVerify: UnrestrictedPerturbationFamily
using JuMP
using Random

include("../../TestHelpers.jl") # Added include for TestHelpers
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
    input = [1.0, -1.0]
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
        # For input [1.0, -1.0], original output is [1.0, -1.0]. Predicted label is 1.
        # Let's try to make label 2 the target.
        target_label = 2
        non_target_label = 1

        for (test_name, margin) in
            [("positive margin", 0.5), ("zero margin", 0.0), ("negative margin", -0.5)]
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
                @test d[:SolveStatus] == MOI.OPTIMAL
                perturbed_output = JuMP.value.(d[:PerturbedInput]) |> nn
                target_logit = perturbed_output[target_label]
                non_target_logit = perturbed_output[non_target_label]
                # We want y_target - y_nontarget >= margin
                @test target_logit - non_target_logit >= margin - 1e-6 # allow for small numerical errors
                # We do not assert the perturbation is zero, only that the margin constraint is satisfied.
            end
        end
    end

    @testset "worst objective" begin
        # Despite the choice of the `UnrestrictedPerturbationFamily`, since 
        # each element of the input is constrained to be in [0, 1]
        # (see `get_perturbation_specific_keys`), the maximum difference is 2.
        target_label = 2
        non_target_label = 1

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
                if is_feasible
                    @test d[:SolveStatus] == MOI.OPTIMAL
                else
                    @test d[:SolveStatus] == MOI.INFEASIBLE
                end
                if is_feasible
                    perturbed_input = JuMP.value.(d[:PerturbedInput])
                    perturbed_output = perturbed_input |> nn
                    target_logit = perturbed_output[target_label]
                    non_target_logit = perturbed_output[non_target_label]
                    objective_value = JuMP.objective_value(d[:Model])
                    @test objective_value == 2
                    @test target_logit - non_target_logit == 2
                end
            end
        end
    end

end
