module BenchmarkRunnerRegression
using Test
include(joinpath(@__DIR__, "..", "benchmark_wk17a_first100.jl"))
@testset "Model-free benchmark results retain their evidence and route" begin
    # A direct certificate has no witness or JuMP model, and its raw termination
    # establishes infeasibility while the displayed status records no solve.
    certificate = Dict{Symbol,Any}(
        :VerificationSource => :backward_bound,
        :SolveStatus => MOI.INFEASIBLE,
        :ObjectiveValue => NaN,
        :ObjectiveBound => NaN,
        :WitnessAvailable => false,
        :WitnessVerified => false,
    )
    certified = benchmark_result_fields(certificate)
    @test certified.solve_status == "NO_SOLVER"
    @test certified.verification_status == "INFEASIBLE"
    @test certified.verification_source == "backward_bound"
    @test certified.semantic_outcome == "certified_no_adversarial_example"
    @test ismissing(certified.objective_value)
    @test ismissing(certified.objective_bound)
    # Corner and PGD successes are checked witnesses, not optimizer calls.
    for source in (:corner, :pgd)
        witness = merge(
            certificate,
            Dict(
                :VerificationSource => source,
                :SolveStatus => MOI.OPTIMAL,
                :ObjectiveValue => 0.0,
                :ObjectiveBound => 0.0,
                :WitnessAvailable => true,
                :WitnessVerified => true,
            ),
        )
        result = benchmark_result_fields(witness)
        @test result.solve_status == "NO_SOLVER"
        @test result.verification_source == string(source)
        @test result.semantic_outcome == "adversarial_example_found_or_best_known"
        @test result.objective_value == 0.0
    end
    # The public original-input skip has no recorded solver termination.
    skipped = Dict{Symbol,Any}(:WitnessAvailable => true, :WitnessVerified => true)
    original = benchmark_result_fields(skipped)
    @test original.solve_status == "SKIPPED_PREDICTED_IN_TARGETED"
    @test original.verification_source == "predicted_input"
    @test original.objective_value == original.objective_bound == 0.0
    @test original.semantic_outcome == "adversarial_example_found_or_best_known"
    fields = model_free_instrumentation_fields()
    @test keys(fields) == INSTRUMENTATION_COLUMNS
    @test fields.main_solve_wall_time_seconds == 0.0
    @test fields.bound_solver_call_count == 0
    @test ismissing(fields.relu_total_count)
end
end
