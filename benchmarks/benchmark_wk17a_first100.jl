using CSV
using DataFrames
using Dates
using HiGHS
using JuMP
using MIPVerify
using Statistics

include(joinpath(@__DIR__, "BenchmarkHelpers.jl"))
using .BenchmarkHelpers

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

function main()
    args = parse_args(ARGS)
    if !haskey(args, "out")
        error("Missing required argument --out <dir>")
    end

    out_dir = abspath(args["out"])
    mkpath(out_dir)

    dependency_snapshot = collect_dependency_snapshot()
    dependency_versions_path = joinpath(out_dir, "dependency_versions.csv")
    write_dependency_snapshot(dependency_versions_path, dependency_snapshot)

    dependency_manifest_path = joinpath(out_dir, "dependency_manifest.toml")
    cp(active_manifest_path(), dependency_manifest_path; force = true)

    julia_version = string(VERSION)
    dependency_snapshot_sha256 = dependency_snapshot_hash(
        dependency_snapshot;
        julia_version = julia_version,
    )

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
        push!(total_times, Float64(d[:TotalTime]))

        if haskey(d, :Model)
            m = d[:Model]
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
        else
            push!(solve_times, 0.0)
            push!(statuses, "SKIPPED_PREDICTED_IN_TARGETED")
            push!(objective_values, missing)
            push!(objective_bounds, missing)
            push!(semantic_outcomes, "skipped_predicted_in_targeted")
        end
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
        median_solve_time_seconds = [median(solve_times)],
        p90_solve_time_seconds = [quantile(solve_times, 0.9)],
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
        julia_version = [julia_version],
        dependency_snapshot_sha256 = [dependency_snapshot_sha256],
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
