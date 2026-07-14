using CSV
using DataFrames

include(joinpath(@__DIR__, "BenchmarkHelpers.jl"))
using .BenchmarkHelpers:
    regression_ratio,
    percent,
    benchmark_schema_version,
    semantic_outcome_schema_version,
    semantic_partition_columns_present,
    semantic_partition_matches,
    semantic_partition_is_complete,
    SEMANTIC_OUTCOME_SCHEMA_VERSION

function parse_args(args::Vector{String})::Dict{String,String}
    parsed = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            @assert i < length(args) "Missing value for --$key"
            parsed[key] = args[i+1]
            i += 2
        else
            i += 1
        end
    end
    return parsed
end

function load_metrics(dir::String)::DataFrame
    path = joinpath(abspath(dir), "benchmark_metrics.csv")
    @assert isfile(path) "Missing metrics file at $path"
    return DataFrame(CSV.File(path))
end

function main()
    args = parse_args(ARGS)
    @assert haskey(args, "baseline") "Missing required argument --baseline <dir>"
    @assert haskey(args, "candidate") "Missing required argument --candidate <dir>"

    max_regression = parse(Float64, get(args, "max-regression", "0.05"))

    baseline = load_metrics(args["baseline"])
    candidate = load_metrics(args["candidate"])

    baseline_benchmark_schema = benchmark_schema_version(baseline)
    candidate_benchmark_schema = benchmark_schema_version(candidate)
    baseline_semantic_schema = semantic_outcome_schema_version(baseline)
    candidate_semantic_schema = semantic_outcome_schema_version(candidate)
    schemas_match =
        baseline_benchmark_schema == candidate_benchmark_schema &&
        baseline_semantic_schema == candidate_semantic_schema
    if !schemas_match
        println(
            "Schema mismatch: baseline=(benchmark=$baseline_benchmark_schema, semantic=$baseline_semantic_schema), " *
            "candidate=(benchmark=$candidate_benchmark_schema, semantic=$candidate_semantic_schema)",
        )
    end

    base_wall = Float64(baseline[1, :wall_clock_seconds])
    cand_wall = Float64(candidate[1, :wall_clock_seconds])
    base_total = Float64(baseline[1, :sum_total_time_seconds])
    cand_total = Float64(candidate[1, :sum_total_time_seconds])

    wall_delta = regression_ratio(base_wall, cand_wall)
    total_delta = regression_ratio(base_total, cand_total)

    println("Metric,baseline,candidate,delta")
    println("wall_clock_seconds,$base_wall,$cand_wall,$(percent(wall_delta))")
    println("sum_total_time_seconds,$base_total,$cand_total,$(percent(total_delta))")

    outcome_cols = [
        "num_skipped_predicted_in_targeted",
        "num_certified_no_adversarial_example",
        "num_adversarial_example_found_or_best_known",
        "num_time_limit_unresolved",
        "num_no_primal_solution_other",
        "num_missing_objective_value",
    ]
    comparable_outcome_cols =
        filter(col -> col in names(baseline) && col in names(candidate), outcome_cols)
    if !isempty(comparable_outcome_cols)
        println()
        println("Outcome and skipped-input counts:")
        for col in comparable_outcome_cols
            base_val = Int(baseline[1, Symbol(col)])
            cand_val = Int(candidate[1, Symbol(col)])
            delta = cand_val - base_val
            println("$col,$base_val,$cand_val,$delta")
        end
    end

    threshold_text = percent(max_regression)

    failures = String[]
    schemas_match || push!(failures, "schema mismatch")
    wall_delta <= max_regression || push!(failures, "wall-clock regression exceeds $threshold_text")
    total_delta <= max_regression ||
        push!(failures, "total-time regression exceeds $threshold_text")
    if !schemas_match
        # Semantic outcomes are only comparable under identical counting rules.
        println("Semantic verdict: SKIPPED (schema mismatch)")
    else
        semantic_ok = semantic_partition_matches(baseline, candidate)
        partition_complete = if baseline_semantic_schema >= SEMANTIC_OUTCOME_SCHEMA_VERSION
            semantic_partition_is_complete(baseline) && semantic_partition_is_complete(candidate)
        else
            true
        end
        if !semantic_partition_columns_present(baseline) ||
           !semantic_partition_columns_present(candidate)
            push!(failures, "semantic outcome columns missing from baseline or candidate")
        elseif !semantic_ok
            push!(failures, "semantic outcome counts differ")
        end
        partition_complete || push!(failures, "semantic partition counts do not sum to num_samples")
        println("Semantic verdict: $(semantic_ok && partition_complete ? "PASS" : "FAIL")")
    end
    passed = isempty(failures)
    if passed
        println("Gate verdict: PASS (max regression: $threshold_text)")
    else
        println("Gate verdict: FAIL ($(join(failures, "; ")))")
    end

    exit(passed ? 0 : 1)
end

main()
