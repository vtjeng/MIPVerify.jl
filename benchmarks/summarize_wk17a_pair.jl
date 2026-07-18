# Compares two benchmark_wk17a_first100.jl output directories sample-by-sample.
#
# Usage:
#   julia --project=benchmarks benchmarks/summarize_wk17a_pair.jl \
#       --baseline /path/to/baseline-out --candidate /path/to/candidate-out
#
# Reports aggregate deltas, per-sample time distributions (including build+tightening
# time, where certification cost lives), semantic-outcome and solve-status changes,
# and objective-value agreement over commonly solved samples.

using CSV
using DataFrames
using Statistics

include(joinpath(@__DIR__, "BenchmarkHelpers.jl"))
using .BenchmarkHelpers: benchmark_objective

function parse_args(args::Vector{String})::Dict{String,String}
    parsed = Dict{String,String}()
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("Unexpected argument $(args[i])")
        i + 1 <= length(args) || error("Missing value for $(args[i])")
        parsed[args[i][3:end]] = args[i+1]
        i += 2
    end
    return parsed
end

function print_distribution(name::String, base::Vector{Float64}, cand::Vector{Float64})
    for (stat, f) in (
        ("mean", mean),
        ("median", median),
        ("p90", v -> quantile(v, 0.9)),
        ("p99", v -> quantile(v, 0.99)),
        ("max", maximum),
    )
        base_value = f(base)
        cand_value = f(cand)
        delta = base_value == 0 ? NaN : (cand_value - base_value) / base_value * 100
        println(
            "$(name)_$(stat)_seconds,$(round(base_value; digits = 3))," *
            "$(round(cand_value; digits = 3)),$(round(delta; digits = 2))%",
        )
    end
end

function main()
    options = parse_args(ARGS)
    baseline_dir = options["baseline"]
    candidate_dir = options["candidate"]

    base = CSV.read(joinpath(baseline_dir, "benchmark_per_sample.csv"), DataFrame)
    cand = CSV.read(joinpath(candidate_dir, "benchmark_per_sample.csv"), DataFrame)
    base_metrics = CSV.read(joinpath(baseline_dir, "benchmark_metrics.csv"), DataFrame)
    cand_metrics = CSV.read(joinpath(candidate_dir, "benchmark_metrics.csv"), DataFrame)
    base_objective = benchmark_objective(base_metrics)
    cand_objective = benchmark_objective(cand_metrics)
    objectives_match = base_objective == cand_objective

    println("adversarial_example_objective,baseline,candidate")
    println("objective,$base_objective,$cand_objective")

    println("metric,baseline,candidate,delta")
    for column in (:wall_clock_seconds, :sum_total_time_seconds, :sum_solve_time_seconds)
        base_value = Float64(base_metrics[1, column])
        cand_value = Float64(cand_metrics[1, column])
        delta = round((cand_value - base_value) / base_value * 100; digits = 2)
        println(
            "$(column),$(round(base_value; digits = 3))," *
            "$(round(cand_value; digits = 3)),$(delta)%",
        )
    end

    print_distribution(
        "total_time",
        Float64.(base.total_time_seconds),
        Float64.(cand.total_time_seconds),
    )
    print_distribution(
        "solve_time",
        Float64.(base.solve_time_seconds),
        Float64.(cand.solve_time_seconds),
    )
    print_distribution(
        "build_and_tightening_time",
        Float64.(base.total_time_seconds .- base.solve_time_seconds),
        Float64.(cand.total_time_seconds .- cand.solve_time_seconds),
    )

    joined = innerjoin(base, cand; on = :sample_index, renamecols = "_baseline" => "_candidate")
    println("samples_joined,$(nrow(joined))")

    outcome_changed =
        filter(r -> r.semantic_outcome_baseline != r.semantic_outcome_candidate, joined)
    println("semantic_outcome_changes,$(nrow(outcome_changed))")
    for r in eachrow(first(outcome_changed, 20))
        println(
            "  sample $(r.sample_index): " *
            "$(r.semantic_outcome_baseline) -> $(r.semantic_outcome_candidate)",
        )
    end

    status_changed = filter(r -> r.solve_status_baseline != r.solve_status_candidate, joined)
    println("solve_status_changes,$(nrow(status_changed))")
    for r in eachrow(first(status_changed, 20))
        println(
            "  sample $(r.sample_index): " *
            "$(r.solve_status_baseline) -> $(r.solve_status_candidate)",
        )
    end

    if objectives_match
        solved_both = filter(
            r ->
                !ismissing(r.objective_value_baseline) &&
                    !ismissing(r.objective_value_candidate),
            joined,
        )
        println("objective_value_comparable_samples,$(nrow(solved_both))")
        if nrow(solved_both) > 0
            diffs =
                abs.(solved_both.objective_value_baseline .- solved_both.objective_value_candidate)
            println("objective_value_max_abs_diff,$(maximum(diffs))")
            println("objective_value_mean_abs_diff,$(mean(diffs))")
            println("objective_value_diff_count_gt_1e-6,$(count(>(1e-6), diffs))")
        end
    else
        println("objective_value_comparison_skipped,benchmark objective mismatch")
    end

    base_hash = string(base_metrics[1, :dependency_snapshot_sha256])
    cand_hash = string(cand_metrics[1, :dependency_snapshot_sha256])
    println("dependency_snapshot_match,$(base_hash == cand_hash)")
end

main()
