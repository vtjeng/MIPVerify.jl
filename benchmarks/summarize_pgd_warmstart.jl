using CSV
using DataFrames
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "BenchmarkHelpers.jl"))
using .BenchmarkHelpers

function numeric_values(rows, column::Symbol)
    column in propertynames(rows) || return Float64[]
    return Float64[value for value in skipmissing(rows[!, column]) if isfinite(value)]
end

function safe_median(rows, column::Symbol)
    values = numeric_values(rows, column)
    return isempty(values) ? missing : median(values)
end

geometric_mean(values) = exp(mean(log, values))

function bootstrap_geometric_mean_interval(values; iterations = 10_000, seed = 20260720)
    isempty(values) && return (missing, missing)
    rng = MersenneTwister(seed)
    estimates = Vector{Float64}(undef, iterations)
    for iteration in 1:iterations
        sample = rand(rng, values, length(values))
        estimates[iteration] = geometric_mean(sample)
    end
    return (quantile(estimates, 0.025), quantile(estimates, 0.975))
end

function treatment_rows_for_sample(rows, sample_index, treatment)
    return filter(
        row -> Int(row.sample_index) == sample_index && string(row.treatment) == treatment,
        rows,
    )
end

function paired_sample_rows(treatments::DataFrame)
    rows = NamedTuple[]
    for sample_index in sort(unique(Int.(treatments.sample_index)))
        cold = treatment_rows_for_sample(treatments, sample_index, "cold")
        full = treatment_rows_for_sample(treatments, sample_index, "pgd_full")
        original_sparse = treatment_rows_for_sample(treatments, sample_index, "original_sparse")
        pgd_sparse = treatment_rows_for_sample(treatments, sample_index, "pgd_sparse")
        nrow(cold) == 0 && continue
        nrow(full) == 0 && continue

        cold_simplex = safe_median(cold, :simplex_iterations)
        full_simplex = safe_median(full, :simplex_iterations)
        cold_nodes = safe_median(cold, :node_count)
        full_nodes = safe_median(full, :node_count)
        cold_solve = safe_median(cold, :main_solve_wall_time_seconds)
        full_solve = safe_median(full, :main_solve_wall_time_seconds)
        cold_total = safe_median(cold, :end_to_end_time_seconds)
        full_total = safe_median(full, :end_to_end_time_seconds)
        original_sparse_simplex = safe_median(original_sparse, :simplex_iterations)
        pgd_sparse_simplex = safe_median(pgd_sparse, :simplex_iterations)
        cold_bound = safe_median(cold, :objective_bound)
        full_bound = safe_median(full, :objective_bound)
        push!(
            rows,
            (
                sample_index = sample_index,
                stratum = string(first(cold.stratum)),
                block_count = min(nrow(cold), nrow(full)),
                cold_simplex_iterations = cold_simplex,
                pgd_full_simplex_iterations = full_simplex,
                simplex_ratio = (full_simplex + 1) / (cold_simplex + 1),
                cold_node_count = cold_nodes,
                pgd_full_node_count = full_nodes,
                node_ratio = (full_nodes + 1) / (cold_nodes + 1),
                cold_solve_seconds = cold_solve,
                pgd_full_solve_seconds = full_solve,
                solve_time_ratio = full_solve / cold_solve,
                cold_end_to_end_seconds = cold_total,
                pgd_full_end_to_end_seconds = full_total,
                end_to_end_ratio = full_total / cold_total,
                cold_objective_bound = cold_bound,
                pgd_full_objective_bound = full_bound,
                objective_bound_delta = full_bound - cold_bound,
                original_sparse_simplex_iterations = original_sparse_simplex,
                pgd_sparse_simplex_iterations = pgd_sparse_simplex,
                sparse_simplex_ratio = (pgd_sparse_simplex + 1) / (original_sparse_simplex + 1),
                cold_unresolved_blocks = count(==("time_limit_unresolved"), cold.outcome),
                pgd_full_unresolved_blocks = count(==("time_limit_unresolved"), full.outcome),
            ),
        )
    end
    return DataFrame(rows)
end

function ratio_summary(per_sample::DataFrame, column::Symbol)
    values = numeric_values(per_sample, column)
    isempty(values) && return (estimate = missing, lower = missing, upper = missing, count = 0)
    lower, upper = bootstrap_geometric_mean_interval(values)
    return (estimate = geometric_mean(values), lower = lower, upper = upper, count = length(values))
end

function format_value(value; digits = 3)
    ismissing(value) && return "n/a"
    return @sprintf("%.*f", digits, Float64(value))
end

function treatment_summary(treatments::DataFrame)
    rows = NamedTuple[]
    for treatment in ["cold", "original_sparse", "pgd_sparse", "original_full", "pgd_full"]
        selected = filter(row -> string(row.treatment) == treatment, treatments)
        nrow(selected) == 0 && continue
        used_count = if :mip_start_used in propertynames(selected)
            count(==(true), skipmissing(selected.mip_start_used))
        else
            0
        end
        push!(
            rows,
            (
                treatment = treatment,
                runs = nrow(selected),
                median_solve_seconds = safe_median(selected, :main_solve_wall_time_seconds),
                median_end_to_end_seconds = safe_median(selected, :end_to_end_time_seconds),
                median_simplex_iterations = safe_median(selected, :simplex_iterations),
                median_node_count = safe_median(selected, :node_count),
                unresolved = count(==("time_limit_unresolved"), selected.outcome),
                certified = count(==("certified_robust"), selected.outcome),
                attacks = count(==("verified_attack"), selected.outcome),
                starts_used = used_count,
            ),
        )
    end
    return DataFrame(rows)
end

function cold_reference_observation(path::Union{Nothing,String}, current::DataFrame)
    isnothing(path) && return (
        status = "not provided",
        flagged_samples = Int[],
        message = "No comparable same-objective historical cold run was provided.",
    )
    reference = CSV.read(path, DataFrame)
    required = [:sample_index, :treatment, :main_solve_wall_time_seconds]
    if !all(column -> column in propertynames(reference), required)
        return (
            status = "not comparable",
            flagged_samples = Int[],
            message = "The supplied historical file does not use the worst-margin warm-start schema/objective.",
        )
    end
    flags = Int[]
    ratios = Float64[]
    for sample_index in
        intersect(unique(Int.(current.sample_index)), unique(Int.(reference.sample_index)))
        fresh = treatment_rows_for_sample(current, sample_index, "cold")
        historical = treatment_rows_for_sample(reference, sample_index, "cold")
        nrow(fresh) == 0 && continue
        nrow(historical) == 0 && continue
        ratio =
            median(fresh.main_solve_wall_time_seconds) /
            median(historical.main_solve_wall_time_seconds)
        push!(ratios, ratio)
        !(0.5 <= ratio <= 2.0) && push!(flags, sample_index)
    end
    isempty(ratios) && return (
        status = "not comparable",
        flagged_samples = Int[],
        message = "The supplied run has no overlapping cold samples.",
    )
    return (
        status = isempty(flags) ? "comparable, no flags" : "comparable, discrepancy flagged",
        flagged_samples = flags,
        message = "Fresh/historical cold median ratios range from $(minimum(ratios)) to $(maximum(ratios)).",
    )
end

function write_report(
    path,
    mode,
    treatments,
    samples,
    per_sample,
    treatment_stats,
    summaries,
    conclusion,
    acceptance_rate,
    cold_reference,
    expected_samples,
    expected_blocks,
)
    open(path, "w") do io
        println(io, "# PGD worst-margin warm-start report")
        println(io)
        println(io, "## Summary")
        println(io)
        println(io, "- Mode: `$mode`")
        println(io, "- Conclusion: **$conclusion**")
        println(io, "- Eligible paired samples: $(nrow(per_sample)) / $expected_samples")
        println(io, "- Required paired blocks per sample: $expected_blocks")
        println(io, "- `pgd_full` start-use rate: $(format_value(100acceptance_rate; digits = 1))%")
        println(
            io,
            "- Peak recorded RSS: $(format_value(maximum(treatments.process_peak_rss_mb); digits = 1)) MiB",
        )
        println(io, "- Cold-reference check: $(cold_reference.status). $(cold_reference.message)")
        if !isempty(cold_reference.flagged_samples)
            println(
                io,
                "- Cold-reference flagged samples: $(join(cold_reference.flagged_samples, ", "))",
            )
        end
        println(io)
        println(io, "## Paired findings")
        println(io)
        println(io, "| Metric (`pgd_full / cold`) | Geometric mean | 95% bootstrap CI | Samples |")
        println(io, "| --- | ---: | ---: | ---: |")
        for (label, summary) in summaries
            println(
                io,
                "| $label | $(format_value(summary.estimate)) | " *
                "[$(format_value(summary.lower)), $(format_value(summary.upper))] | $(summary.count) |",
            )
        end
        println(io)
        println(
            io,
            "Negative final-bound deltas favor `pgd_full`. Median delta: ",
            format_value(median(per_sample.objective_bound_delta)),
            ".",
        )
        println(io)
        println(io, "## Treatment-level observations")
        println(io)
        println(
            io,
            "| Treatment | Runs | Median solve (s) | Median end-to-end (s) | Median simplex | Median nodes | Unresolved | Certified | Attacks | Starts used |",
        )
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in eachrow(treatment_stats)
            println(
                io,
                "| $(row.treatment) | $(row.runs) | $(format_value(row.median_solve_seconds)) | " *
                "$(format_value(row.median_end_to_end_seconds)) | " *
                "$(format_value(row.median_simplex_iterations; digits = 0)) | " *
                "$(format_value(row.median_node_count; digits = 0)) | $(row.unresolved) | " *
                "$(row.certified) | $(row.attacks) | $(row.starts_used) |",
            )
        end
        println(io)
        println(io, "## Method")
        println(io)
        selected_treatments = join(treatment_stats.treatment, ", ")
        println(
            io,
            "Each sample uses one tightened base model. The selected treatments ($selected_treatments) " *
            "are copied from that base and run serially in a block-rotated order. Per-sample values are medians across " *
            "blocks; aggregate ratios are geometric means across samples. The simplex ratio uses " *
            "`(pgd_full + 1) / (cold + 1)`.",
        )
        println(io)
        println(io, "## Limitations")
        println(io)
        println(
            io,
            "Wall time is noisy, so simplex iterations are primary and nodes/bound trajectories " *
            "are corroborating. Time-limited pairs are retained. PGD and full-start completion are " *
            "charged in end-to-end time. Historical feasibility-objective runs are not comparable " *
            "to this exact worst-margin objective.",
        )
        if mode == "calibration"
            println(
                io,
                "This is a calibration result, not cohort-level evidence; it only determines " *
                "whether the full start is accepted and the main cohort should proceed.",
            )
        end
        println(io)
        println(io, "## Per-sample effects")
        println(io)
        println(io, "See `per_sample_effects.csv` for the complete paired table.")
        println(io)
        println(
            io,
            "Sample metadata, PGD margins, formulation cost, and completion cost are in ",
            "`benchmark_samples.csv` ($(nrow(samples)) rows).",
        )
    end
end

function main()
    args = parse_args(ARGS)
    run_dir = get(args, "run", "")
    isempty(run_dir) && error("--run is required")
    output_dir = get(args, "out", joinpath(run_dir, "analysis"))
    mode = get(args, "mode", "cohort")
    mode in ("calibration", "cohort") || error("--mode must be calibration or cohort")
    expected_samples = parse(Int, get(args, "expected-samples", mode == "cohort" ? "12" : "1"))
    expected_blocks = parse(Int, get(args, "expected-blocks", "3"))
    reference_path = get(args, "cold-reference", nothing)

    treatments = CSV.read(joinpath(run_dir, "benchmark_per_treatment.csv"), DataFrame)
    samples = CSV.read(joinpath(run_dir, "benchmark_samples.csv"), DataFrame)
    per_sample = paired_sample_rows(treatments)
    treatment_stats = treatment_summary(treatments)
    mkpath(output_dir)
    atomic_write_csv = (path, table) -> begin
        temporary_path = path * ".tmp"
        CSV.write(temporary_path, table)
        mv(temporary_path, path; force = true)
    end
    atomic_write_csv(joinpath(output_dir, "per_sample_effects.csv"), per_sample)
    atomic_write_csv(joinpath(output_dir, "treatment_summary.csv"), treatment_stats)

    simplex = ratio_summary(per_sample, :simplex_ratio)
    nodes = ratio_summary(per_sample, :node_ratio)
    solve_time = ratio_summary(per_sample, :solve_time_ratio)
    end_to_end = ratio_summary(per_sample, :end_to_end_ratio)
    sparse_simplex = ratio_summary(per_sample, :sparse_simplex_ratio)
    summaries = [
        "Simplex iterations" => simplex,
        "Node count" => nodes,
        "Main solve time" => solve_time,
        "End-to-end time" => end_to_end,
        "PGD sparse / original sparse simplex" => sparse_simplex,
    ]

    full_rows = filter(row -> string(row.treatment) == "pgd_full", treatments)
    acceptance_values = collect(skipmissing(full_rows.mip_start_used))
    acceptance_rate =
        isempty(acceptance_values) ? 0.0 :
        count(==(true), acceptance_values) / length(acceptance_values)
    complete_pairs =
        nrow(per_sample) == expected_samples && all(==(expected_blocks), per_sample.block_count)
    fixed_limit_favors_cold =
        sum(per_sample.pgd_full_unresolved_blocks) > sum(per_sample.cold_unresolved_blocks) ||
        median(per_sample.objective_bound_delta) > 0
    conclusion = if mode == "calibration"
        acceptance_rate == 1.0 && complete_pairs ? "calibration passed" : "calibration failed"
    elseif !complete_pairs
        "incomplete cohort"
    elseif acceptance_rate < 0.9 || simplex.estimate > 0.90 || fixed_limit_favors_cold
        "not supported"
    elseif simplex.upper < 1.0
        "solver-level support"
    else
        "promising but inconclusive"
    end
    cold_reference = cold_reference_observation(reference_path, treatments)

    summary = DataFrame(
        mode = [mode],
        conclusion = [conclusion],
        paired_samples = [nrow(per_sample)],
        expected_samples = [expected_samples],
        expected_blocks = [expected_blocks],
        complete_pairs = [complete_pairs],
        full_start_acceptance_rate = [acceptance_rate],
        simplex_ratio = [simplex.estimate],
        simplex_ratio_ci_lower = [simplex.lower],
        simplex_ratio_ci_upper = [simplex.upper],
        node_ratio = [nodes.estimate],
        solve_time_ratio = [solve_time.estimate],
        end_to_end_ratio = [end_to_end.estimate],
        sparse_simplex_ratio = [sparse_simplex.estimate],
        fixed_limit_favors_cold = [fixed_limit_favors_cold],
        cold_reference_status = [cold_reference.status],
        peak_rss_mb = [maximum(treatments.process_peak_rss_mb)],
    )
    atomic_write_csv(joinpath(output_dir, "summary.csv"), summary)
    write_report(
        joinpath(output_dir, "report.md"),
        mode,
        treatments,
        samples,
        per_sample,
        treatment_stats,
        summaries,
        conclusion,
        acceptance_rate,
        cold_reference,
        expected_samples,
        expected_blocks,
    )
    println("$conclusion; report=$(joinpath(output_dir, "report.md"))")
end

main()
