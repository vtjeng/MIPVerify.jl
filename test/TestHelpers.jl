module TestHelpers

using Base.Test
using JuMP
using MathProgBase

using MIPVerify
using MIPVerify: find_adversarial_example
using MIPVerify: NeuralNet
using MIPVerify: PerturbationFamily

const TEST_DEFAULT_TIGHTENING_ALGORITHM = lp

if Pkg.installed("Gurobi") == nothing
    using Cbc
    main_solver = CbcSolver(logLevel=0)
    tightening_solver = CbcSolver(logLevel=0, seconds=20)
else
    using Gurobi
    main_solver = GurobiSolver(Gurobi.Env())
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=20)
end

function get_main_solver()::MathProgBase.SolverInterface.AbstractMathProgSolver
    main_solver
end

function get_tightening_solver()::MathProgBase.SolverInterface.AbstractMathProgSolver
    tightening_solver
end

function get_new_model()::Model
    solver = get_main_solver()
    MathProgBase.setparameters!(solver, Silent = true)
    return Model(solver=solver)
end

"""
Tests the `find_adversarial_example` function.
  - If `x0` is already classified in the target label, `expected_objective_value`
    should be 0.
  - If there is no adversarial example for the specified parameters, 
    `expected_objective_value` should be NaN.
  - If there is an adversarial example, checks that the objective value is as expected,
    and that the perturbed output for the target label exceeds the perturbed 
    output for all other labels by `tolerance`.
"""
function test_find_adversarial_example(
    nn::NeuralNet, 
    x0::Array{<:Real, N}, 
    target_selection::Union{Integer, Array{<:Integer, 1}}, 
    pp::PerturbationFamily, 
    norm_order::Real,
    tolerance::Real, 
    expected_objective_value::Real,
    ) where {N} 
    d = find_adversarial_example(
        nn, x0, target_selection, get_main_solver(),
        pp = pp, norm_order = norm_order, tolerance = tolerance, rebuild=false, 
        tightening_solver=get_tightening_solver(), tightening_algorithm=TEST_DEFAULT_TIGHTENING_ALGORITHM)
    println(d[:SolveStatus])
    if d[:SolveStatus] == :Infeasible || d[:SolveStatus] == :InfeasibleOrUnbounded
        @test isnan(expected_objective_value)
    else
        actual_objective_value = getobjectivevalue(d[:Model])
        if expected_objective_value == 0
            @test isapprox(actual_objective_value, expected_objective_value; atol=1e-4)
        else
            @test isapprox(actual_objective_value, expected_objective_value; rtol=5e-5)
            
            perturbed_output = getvalue(d[:PerturbedInput]) |> nn
            perturbed_target_output = maximum(perturbed_output[Bool[i∈d[:TargetIndexes] for i = 1:length(d[:Output])]])
            maximum_perturbed_other_output = maximum(perturbed_output[Bool[i∉d[:TargetIndexes] for i = 1:length(d[:Output])]])
            @test perturbed_target_output/(maximum_perturbed_other_output+tolerance)≈1 atol=5e-5
        end
    end
end

"""
Runs tests on the neural net described by `nn` for input `x0` and the objective values
indicated in `expected objective values`.

# Arguments
- `expected_objective_values::Dict`: 
   d[(target_selection, perturbation_parameter, norm_order, tolerance)] = expected_objective_value
   `expected_objective_value` is `NaN` if there is no perturbation that brings the image
   into the target category.
"""
function batch_test_adversarial_example(
    nn::NeuralNet, 
    x0::Array{<:Real, N},
    expected_objective_values::Dict
) where {N}
    for (test_params, expected_objective_value) in expected_objective_values
        (target_selection, pp, norm_order, tolerance) = test_params
        @testset "target label = $target_selection, $(string(pp)) perturbation, norm order = $norm_order, tolerance = $tolerance" begin
            test_find_adversarial_example(
                nn, x0, 
                target_selection, pp, norm_order, tolerance, expected_objective_value)
            end
        end
    end
end