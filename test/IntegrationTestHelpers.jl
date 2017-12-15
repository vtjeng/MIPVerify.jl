module IntegrationTestHelpers

using JuMP

using MIPVerify: find_adversarial_example
using MIPVerify: NeuralNetParameters
using MIPVerify: PerturbationParameters
using Base.Test
using Gurobi
using Cbc

"""
    test_find_adversarial_example(nnparams, x0, target_label, pp, norm_order, tolerance, expected_objective_value, solver_type)

Tests the `find_adversarial_example` function.
  - If `x0` is already classified in the target label, `expected_objective_value`
    should be 0.
  - If there is no adversarial example for the specified parameters, `expected_objective_value`
    should be NaN.
  - If there is an adversarial example, checks that the objective value is as expected,
    and that the perturbed output for the target label exceeds the perturbed 
    output for all other labels by `tolerance`.
"""
function test_find_adversarial_example(
    nnparams::NeuralNetParameters, 
    x0::Array{T, N}, 
    target_label::Int, 
    pp::PerturbationParameters, 
    norm_order::Real,
    tolerance::Real, 
    expected_objective_value::Real,
    solver_type::DataType,
    ) where {T<:Real, N} 
    d = find_adversarial_example(
        nnparams, x0, target_label, solver_type, 
        pp = pp, norm_order = norm_order, tolerance = tolerance, rebuild=false)
    if d[:SolveStatus] == :Infeasible
        @test isnan(expected_objective_value)
    else
        if expected_objective_value == 0
            @test getobjectivevalue(d[:Model]) == 0
        else
            actual_objective_value = getobjectivevalue(d[:Model])
            println("Actual objective value: $actual_objective_value")
            @test actual_objective_value/expected_objective_value≈1 atol=5e-5
            
            perturbed_output = getvalue(d[:PerturbedInput]) |> nnparams
            perturbed_target_output = perturbed_output[target_label]
            maximum_perturbed_other_output = maximum(perturbed_output[1:end .!= target_label])
            @test perturbed_target_output/(maximum_perturbed_other_output+tolerance)≈1 atol=5e-5
        end
    end
end

"""
    batch_test_adversarial_example(nnparams, x0, expected_objective_values)

Runs tests on the neural net described by `nnparams` for input `x0` and the objective values
indicated in `expected objective values`.

# Arguments
- `expected_objective_values::Dict`: 
   d[target_label][perturbation_parameter][norm_order][tolerance] = expected_objective_value
   `expected_objective_value` is `NaN` if model 
"""
function batch_test_adversarial_example(
    nnparams::NeuralNetParameters, 
    x0::Array{T, N},
    expected_objective_values::Dict{Int, Dict{PerturbationParameters, Dict{Real, Dict{Real, Float64}}}}
) where {T<:Real, N}
    for (target_label, d0) in expected_objective_values
        @testset "target label = $target_label" begin
        for (pp, d1) in d0
            @testset "$(string(pp)) perturbation" begin
            for (norm_order, d2) in d1
                @testset "norm order = $norm_order" begin
                for (tolerance, expected_objective_value) in d2
                    @testset "tolerance = $tolerance" begin
                    test_find_adversarial_example(
                        nnparams, x0, 
                        target_label, pp, norm_order, tolerance, expected_objective_value,
                        GurobiSolver)
                    end
                    println("Completed target label = $target_label, $(string(pp)) perturbation, norm order = $norm_order, tolerance = $tolerance.")
                end
                end
            end
            end
        end
        end
    end
end

end