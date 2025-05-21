using Test
using MIPVerify
using JuMP
# Random is not strictly needed for the current static tests but useful for variations
# using Random 

# Assuming TestHelpers.jl is in the same directory or accessible via the load path set up by runtests.jl
include("../../TestHelpers.jl") # For get_test_solver()

@testset "Margin Parameter Tests" begin

    # Define a simple neural network for testing
    # Network: Input (2) -> Linear(2->3) -> ReLU -> Linear(3->2) -> Output (2)
    
    # Common solver and options
    solver = get_test_solver()
    main_solve_options = Dict("output_flag" => false) # Suppress solver output

    @testset "AdversarialExampleObjective: closest" begin
        # Network tailored for 'closest' objective tests
        # Layer 1: Linear
        W1_c = [1.0 1.0; -1.0 -1.0; 0.5 -0.5] 
        b1_c = [0.0, 0.0, 0.0]
        layer1_c = MIPVerify.Linear(W1_c, b1_c)
        # Layer 2: ReLU
        layer2_c = MIPVerify.ReLU()
        # Layer 3: Linear - Adjusted to make output logits potentially close
        W3_c = [1.0 -1.0 0.9;  # Logit for class 1
                1.0 -1.0 1.0]  # Logit for class 2 (target for one test)
        b3_c = [0.0, 0.0]
        layer3_c = MIPVerify.Linear(W3_c, b3_c)
        nn_closest = MIPVerify.Sequential([layer1_c, layer2_c, layer3_c], "SimpleClosestMarginNet")

        # Input: [1.0, 0.0]
        # l1_out = W1_c*input + b1_c = [1, -1, 0.5]
        # l2_out (ReLU) = [1, 0, 0.5]
        # l3_out (output) = W3_c*l2_out + b3_c 
        #   = [1*1 - 0*1 + 0.5*0.9, 1*1 - 0*1 + 0.5*1.0]
        #   = [1 + 0.45, 1 + 0.5] = [1.45, 1.5]
        # Original prediction is index 2 (1.5 > 1.45)
        input_c = [1.0, 0.0] 
        target_selection_c = 1 # Try to make it class 1

        # Scenario 1: Margin makes a difference
        margin_val_c = 0.2 # A margin that should force a change

        # Test with margin = 0 (baseline)
        result_no_margin = MIPVerify.find_adversarial_example(
            nn_closest, input_c, target_selection_c, solver, main_solve_options,
            norm_order=1, adversarial_example_objective=MIPVerify.closest, margin=0.0,
            pp=MIPVerify.UnrestrictedPerturbationFamily(), solve_if_predicted_in_targeted=true
        )

        @test result_no_margin[:SolveStatus] == MIPVerify.Optimal || result_no_margin[:SolveStatus] == MIPVerify.Feasible
        if result_no_margin[:SolveStatus] == MIPVerify.Optimal || result_no_margin[:SolveStatus] == MIPVerify.Feasible
            output_no_margin = result_no_margin[:Output]
            perturbed_input_no_margin = result_no_margin[:PerturbedInput]

            @test input_c != perturbed_input_no_margin
            @test MIPVerify.get_max_index(output_no_margin) == target_selection_c
            
            # Logit difference for margin=0 case
            other_index_c = (target_selection_c == 1) ? 2 : 1
            diff_no_margin = output_no_margin[target_selection_c] - output_no_margin[other_index_c]
            @test diff_no_margin >= 0.0 - 1e-4 # Should be at least 0

            # Test with margin
            result_with_margin = MIPVerify.find_adversarial_example(
                nn_closest, input_c, target_selection_c, solver, main_solve_options,
                norm_order=1, adversarial_example_objective=MIPVerify.closest, margin=margin_val_c,
                pp=MIPVerify.UnrestrictedPerturbationFamily(), solve_if_predicted_in_targeted=true
            )

            @test result_with_margin[:SolveStatus] == MIPVerify.Optimal || result_with_margin[:SolveStatus] == MIPVerify.Feasible
            if result_with_margin[:SolveStatus] == MIPVerify.Optimal || result_with_margin[:SolveStatus] == MIPVerify.Feasible
                perturbed_input_margin = result_with_margin[:PerturbedInput]
                output_margin = result_with_margin[:Output]
                @test input_c != perturbed_input_margin
                @test MIPVerify.get_max_index(output_margin) == target_selection_c
                
                diff_with_margin = output_margin[target_selection_c] - output_margin[other_index_c]
                @test diff_with_margin >= margin_val_c - 1e-4

                # Crucial Check: The margin forced a larger difference or a different perturbation.
                # We expect that the difference achieved with margin=0 is LESS than margin_val_c.
                # This demonstrates the margin is actively constraining the solution.
                @test diff_no_margin < margin_val_c # This is key to show margin is effective

                # Perturbation norm might be larger with margin
                norm_with_margin = sum(abs.(result_with_margin[:Perturbation]))
                norm_without_margin = sum(abs.(result_no_margin[:Perturbation]))
                @test norm_with_margin >= norm_without_margin - 1e-4
            end
        end

        # Scenario 2: No adversarial example found with a large margin
        large_margin_c = 5.0 
        result_large_margin_c = MIPVerify.find_adversarial_example(
            nn_closest, input_c, target_selection_c, solver, main_solve_options,
            norm_order=1, adversarial_example_objective=MIPVerify.closest, margin=large_margin_c,
            pp=MIPVerify.UnrestrictedPerturbationFamily(), solve_if_predicted_in_targeted=true
        )
        @test result_large_margin_c[:SolveStatus] == MIPVerify.Infeasible || result_large_margin_c[:SolveStatus] == MIPVerify.InfeasibleOrUnbounded
    end

    @testset "AdversarialExampleObjective: worst" begin
        # Network for 'worst' objective tests (can be same as 'closest' or different)
        W1_w = [1.0 1.0; -1.0 -1.0; 0.5 -0.5]
        b1_w = [0.0, 0.0, 0.0]
        layer1_w = MIPVerify.Linear(W1_w, b1_w)
        layer2_w = MIPVerify.ReLU()
        W3_w = [1.0 -1.0 0.5; -0.5 1.0 -1.0]
        b3_w = [0.0, 0.0]
        layer3_w = MIPVerify.Linear(W3_w, b3_w)
        nn_worst = MIPVerify.Sequential([layer1_w, layer2_w, layer3_w], "SimpleWorstMarginNet")

        input_w = [0.5, 0.5] 
        # nn_worst(input_w) before perturbation:
        # l1_out = W1_w*input + b1_w = [1, -1, 0]
        # l2_out (ReLU) = [1, 0, 0]
        # l3_out (output) = W3_w*l2_out + b3_w = [1, -0.5]
        # Original prediction is index 1 (1 > -0.5)

        target_selection_w = 2 # Try to make it class 2
        margin_val_w = 0.2

        # Scenario 1: Margin influences the 'worst' objective value
        result_worst_margin = MIPVerify.find_adversarial_example(
            nn_worst, input_w, target_selection_w, solver, main_solve_options,
            norm_order=Inf, adversarial_example_objective=MIPVerify.worst, margin=margin_val_w,
            pp=MIPVerify.UnrestrictedPerturbationFamily(), solve_if_predicted_in_targeted=true
        )
        
        @test result_worst_margin[:SolveStatus] == MIPVerify.Optimal || result_worst_margin[:SolveStatus] == MIPVerify.Feasible
        if result_worst_margin[:SolveStatus] == MIPVerify.Optimal || result_worst_margin[:SolveStatus] == MIPVerify.Feasible
            perturbed_input_worst = result_worst_margin[:PerturbedInput]
            output_worst = result_worst_margin[:Output]
            @test input_w != perturbed_input_worst
            @test MIPVerify.get_max_index(output_worst) == target_selection_w
            
            other_index_w = (target_selection_w == 1) ? 2 : 1
            actual_diff_w = output_worst[target_selection_w] - output_worst[other_index_w]
            
            # For 'worst' objective, the objective is to maximize this difference (v_obj).
            # The constraint is that this difference must be >= margin.
            @test actual_diff_w >= margin_val_w - 1e-4 
            
            # If :Model is part of the output and populated, we could check JuMP.objective_value
            # For example: if haskey(result_worst_margin, :Model) && !isnothing(result_worst_margin[:Model].internalModel)
            #   obj_val = JuMP.objective_value(result_worst_margin[:Model])
            #   @test obj_val ≈ actual_diff_w atol=1e-4
            #   # And obj_val itself should be >= margin_val_w, but this is implicitly tested by actual_diff_w >= margin_val_w
            # end
        end

        # Scenario 2: No solution with large margin for 'worst'
        large_margin_worst = 10.0 
        result_large_margin_worst = MIPVerify.find_adversarial_example(
            nn_worst, input_w, target_selection_w, solver, main_solve_options,
            norm_order=Inf, adversarial_example_objective=MIPVerify.worst, margin=large_margin_worst,
            pp=MIPVerify.UnrestrictedPerturbationFamily(), solve_if_predicted_in_targeted=true
        )
        @test result_large_margin_worst[:SolveStatus] == MIPVerify.Infeasible || result_large_margin_worst[:SolveStatus] == MIPVerify.InfeasibleOrUnbounded
    end
end # @testset "Margin Parameter Tests"
