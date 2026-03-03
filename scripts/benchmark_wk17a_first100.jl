using CSV
using DataFrames
using Dates
using HiGHS
using JuMP
using MIPVerify

function parse_args(args::Vector{String})::Dict{String,String}
    parsed = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            if i == length(args) || startswith(args[i+1], "--")
                parsed[key] = "true"
                i += 1
            else
                parsed[key] = args[i+1]
                i += 2
            end
        else
            i += 1
        end
    end
    return parsed
end

function parse_sample_spec(spec::String)::Vector{Int}
    if occursin(":", spec)
        parts = split(spec, ":")
        @assert length(parts) in (2, 3) "Sample spec must be start:stop or start:step:stop."
        if length(parts) == 2
            start_idx = parse(Int, parts[1])
            stop_idx = parse(Int, parts[2])
            return collect(start_idx:stop_idx)
        else
            start_idx = parse(Int, parts[1])
            step = parse(Int, parts[2])
            stop_idx = parse(Int, parts[3])
            return collect(start_idx:step:stop_idx)
        end
    end
    return parse.(Int, split(spec, ","))
end

function get_optimizer_main_and_tightening_options(main_time_limit::Float64)
    optimizer = HiGHS.Optimizer
    main_solve_options = Dict("output_flag" => false, "time_limit" => main_time_limit)
    tightening_options = Dict("output_flag" => false, "time_limit" => 20.0)
    return (optimizer, main_solve_options, tightening_options)
end

function parse_tightening_algorithm(name::String)::MIPVerify.TighteningAlgorithm
    lowered = lowercase(strip(name))
    if lowered == "interval_arithmetic"
        return MIPVerify.interval_arithmetic
    elseif lowered == "lp"
        return MIPVerify.lp
    elseif lowered == "mip"
        return MIPVerify.mip
    else
        error("Unsupported tightening algorithm $name")
    end
end

function safe_sum(xs)::Float64
    return sum(filter(isfinite, xs))
end

function is_infeasible_status(status::String)::Bool
    return status == "INFEASIBLE" || status == "INFEASIBLE_OR_UNBOUNDED"
end

function classify_semantic_outcome(status::String, objective_value::Union{Missing,Float64})::String
    if is_infeasible_status(status)
        return "certified_no_adversarial_example"
    elseif !ismissing(objective_value)
        return "adversarial_example_found_or_best_known"
    elseif status == "TIME_LIMIT"
        return "time_limit_unresolved"
    else
        return "no_primal_solution_other"
    end
end

function maybe_parse_norm_order(raw::String)
    lowered = lowercase(strip(raw))
    if lowered == "inf"
        return Inf
    end
    return parse(Float64, raw)
end

function main()
    args = parse_args(ARGS)
    if !haskey(args, "out")
        error("Missing required argument --out <dir>")
    end

    out_dir = abspath(args["out"])
    mkpath(out_dir)

    sample_spec = get(args, "samples", "1:100")
    sample_indices = parse_sample_spec(sample_spec)
    tightening_algorithm = parse_tightening_algorithm(get(args, "tightening", "mip"))
    main_time_limit = parse(Float64, get(args, "main-time-limit", "120"))
    norm_order = maybe_parse_norm_order(get(args, "norm-order", "Inf"))

    (optimizer, main_solve_options, tightening_options) =
        get_optimizer_main_and_tightening_options(main_time_limit)

    MIPVerify.set_log_level!(get(args, "log-level", "warn"))
    dataset = MIPVerify.read_datasets("MNIST")
    nn = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1)

    started_at = now()
    wall_start = time()
    total_times = Float64[]
    solve_times = Float64[]
    statuses = String[]
    semantic_outcomes = String[]
    objective_values = Union{Missing,Float64}[]
    objective_bounds = Union{Missing,Float64}[]
    for sample_index in sample_indices
        input = MIPVerify.get_image(dataset.test, sample_index)
        true_label = MIPVerify.get_label(dataset.test, sample_index) + 1
        d = MIPVerify.find_adversarial_example(
            nn,
            input,
            true_label,
            optimizer,
            main_solve_options,
            invert_target_selection = true,
            pp = pp,
            norm_order = norm_order,
            tightening_algorithm = tightening_algorithm,
            tightening_options = tightening_options,
            solve_if_predicted_in_targeted = false,
        )
        m = d[:Model]
        push!(total_times, Float64(d[:TotalTime]))
        push!(solve_times, Float64(d[:SolveTime]))
        push!(statuses, string(d[:SolveStatus]))

        objective_value = try
            Float64(JuMP.objective_value(m))
        catch
            missing
        end
        objective_bound = try
            Float64(JuMP.objective_bound(m))
        catch
            missing
        end
        push!(objective_values, objective_value)
        push!(objective_bounds, objective_bound)
        push!(semantic_outcomes, classify_semantic_outcome(statuses[end], objective_value))
    end
    wall_clock_seconds = time() - wall_start
    completed_at = now()

    per_sample = DataFrame(
        sample_index = sample_indices,
        solve_status = statuses,
        semantic_outcome = semantic_outcomes,
        total_time_seconds = total_times,
        solve_time_seconds = solve_times,
        objective_value = objective_values,
        objective_bound = objective_bounds,
    )
    per_sample_path = joinpath(out_dir, "benchmark_per_sample.csv")
    CSV.write(per_sample_path, per_sample)

    status_counts = Dict{String,Int}()
    for status in statuses
        status_counts[status] = get(status_counts, status, 0) + 1
    end
    semantic_counts = Dict{String,Int}()
    for semantic in semantic_outcomes
        semantic_counts[semantic] = get(semantic_counts, semantic, 0) + 1
    end

    metrics = DataFrame(
        wall_clock_seconds = [wall_clock_seconds],
        sum_total_time_seconds = [safe_sum(total_times)],
        sum_solve_time_seconds = [safe_sum(solve_times)],
        num_samples = [length(sample_indices)],
        num_rows_in_summary = [nrow(per_sample)],
        num_optimal_status = [get(status_counts, "OPTIMAL", 0)],
        num_infeasible_status = [get(status_counts, "INFEASIBLE", 0)],
        num_infeasible_or_unbounded_status = [get(status_counts, "INFEASIBLE_OR_UNBOUNDED", 0)],
        num_time_limit_status = [get(status_counts, "TIME_LIMIT", 0)],
        num_certified_no_adversarial_example = [
            get(semantic_counts, "certified_no_adversarial_example", 0),
        ],
        num_adversarial_example_found_or_best_known = [
            get(semantic_counts, "adversarial_example_found_or_best_known", 0),
        ],
        num_time_limit_unresolved = [get(semantic_counts, "time_limit_unresolved", 0)],
        num_no_primal_solution_other = [get(semantic_counts, "no_primal_solution_other", 0)],
        num_missing_objective_value = [sum(ismissing.(objective_values))],
        tightening_algorithm = [string(tightening_algorithm)],
        norm_order = [string(norm_order)],
        main_time_limit_seconds = [main_time_limit],
        started_at = [string(started_at)],
        completed_at = [string(completed_at)],
        julia_threads = [Threads.nthreads()],
        per_sample_path = [per_sample_path],
    )

    metrics_path = joinpath(out_dir, "benchmark_metrics.csv")
    CSV.write(metrics_path, metrics)
    println("Semantic summary:")
    println(
        "  certified_no_adversarial_example=$(metrics[1, :num_certified_no_adversarial_example])",
    )
    println(
        "  adversarial_example_found_or_best_known=$(metrics[1, :num_adversarial_example_found_or_best_known])",
    )
    println("  time_limit_unresolved=$(metrics[1, :num_time_limit_unresolved])")
    println("  no_primal_solution_other=$(metrics[1, :num_no_primal_solution_other])")
    println("Wrote metrics to $metrics_path")
end

main()
