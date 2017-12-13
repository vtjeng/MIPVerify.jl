module IntegrationTestHelpers

using JuMP

using MIPVerify: find_adversarial_example
using MIPVerify: NeuralNetParameters
using MIPVerify: PerturbationParameters
using Base.Test
using Gurobi

function test_find_adversarial_example(
    nnparams::NeuralNetParameters, 
    x0::Array{T, N}, 
    target_label::Int, 
    solver_type::DataType,
    pp::PerturbationParameters, 
    norm_order::Real,
    tolerance::Real, 
    expected_objective_value::Real) where {T<:Real, N} 
    d = find_adversarial_example(
        nnparams, x0, target_label, solver_type, 
        pp = pp, norm_order = norm_order, tolerance = tolerance, rebuild=true)
    if d[:SolveStatus] == :Infeasible
        @test isnan(expected_objective_value)
    else
        @test getobjectivevalue(d[:Model]) ≈ expected_objective_value
        perturbed_output = getvalue(d[:PerturbedInput]) |> nnparams
        @test perturbed_output[target_label] ≈ maximum(perturbed_output[1:end .!= target_label]) + tolerance
    end
end

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
                        nnparams, x0, target_label, 
                        GurobiSolver, 
                        pp, norm_order, tolerance, expected_objective_value)
                    end
                end
                end
            end
            end
        end
        end
    end
end

end