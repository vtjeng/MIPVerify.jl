using CSV
using DataFrames

include(joinpath(@__DIR__, "BenchmarkHelpers.jl"))
using .BenchmarkHelpers: regression_ratio, percent

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

    base_wall = Float64(baseline[1, :wall_clock_seconds])
    cand_wall = Float64(candidate[1, :wall_clock_seconds])
    base_total = Float64(baseline[1, :sum_total_time_seconds])
    cand_total = Float64(candidate[1, :sum_total_time_seconds])

    wall_delta = regression_ratio(base_wall, cand_wall)
    total_delta = regression_ratio(base_total, cand_total)

    println("Metric,baseline,candidate,delta")
    println("wall_clock_seconds,$base_wall,$cand_wall,$(percent(wall_delta))")
    println("sum_total_time_seconds,$base_total,$cand_total,$(percent(total_delta))")

    semantic_cols = [
        "num_certified_no_adversarial_example",
        "num_adversarial_example_found_or_best_known",
        "num_time_limit_unresolved",
        "num_no_primal_solution_other",
        "num_missing_objective_value",
    ]
    if all(col -> col in names(baseline) && col in names(candidate), semantic_cols)
        println()
        println("Semantic outcomes (counts):")
        for col in semantic_cols
            base_val = Int(baseline[1, Symbol(col)])
            cand_val = Int(candidate[1, Symbol(col)])
            delta = cand_val - base_val
            println("$col,$base_val,$cand_val,$delta")
        end
    end

    wall_ok = wall_delta <= max_regression
    total_ok = total_delta <= max_regression
    passed = wall_ok && total_ok
    threshold_text = percent(max_regression)
    verdict = passed ? "PASS" : "FAIL"
    println("Gate verdict: $verdict (max regression: $threshold_text)")

    exit(passed ? 0 : 1)
end

main()
