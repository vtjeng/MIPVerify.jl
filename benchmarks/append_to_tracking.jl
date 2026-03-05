using CSV
using DataFrames
using Statistics

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

function main()
    args = parse_args(ARGS)

    metrics_csv = args["metrics-csv"]
    per_sample_csv = args["per-sample-csv"]
    tracking_csv = args["tracking-csv"]
    date = args["date"]
    commit_sha = args["commit-sha"]

    metrics = CSV.read(metrics_csv, DataFrame)
    per_sample = CSV.read(per_sample_csv, DataFrame)

    # Compute median and p90 solve time from per-sample data
    solve_times = per_sample.solve_time_seconds
    median_solve_time = median(solve_times)
    p90_solve_time = quantile(solve_times, 0.9)

    # Build the tracking row
    row = DataFrame(
        date = [date],
        commit_sha = [commit_sha],
        wall_clock_seconds = [metrics[1, :wall_clock_seconds]],
        sum_total_time_seconds = [metrics[1, :sum_total_time_seconds]],
        sum_solve_time_seconds = [metrics[1, :sum_solve_time_seconds]],
        median_solve_time_seconds = [median_solve_time],
        p90_solve_time_seconds = [p90_solve_time],
        num_samples = [metrics[1, :num_samples]],
        num_certified_no_adversarial_example = [
            metrics[1, :num_certified_no_adversarial_example],
        ],
        num_adversarial_example_found_or_best_known = [
            metrics[1, :num_adversarial_example_found_or_best_known],
        ],
        num_time_limit_unresolved = [metrics[1, :num_time_limit_unresolved]],
        num_no_primal_solution_other = [metrics[1, :num_no_primal_solution_other]],
    )

    # Create or append to tracking CSV
    if isfile(tracking_csv)
        existing = CSV.read(tracking_csv, DataFrame)
        combined = vcat(existing, row)
        CSV.write(tracking_csv, combined)
    else
        CSV.write(tracking_csv, row)
    end

    println("Appended tracking row for $(date) ($(commit_sha[1:7]))")
end

main()
