using CSV
using DataFrames
using Dates
using MIPVerify

include(joinpath(@__DIR__, "BenchmarkHelpers.jl"))
using .BenchmarkHelpers
include(joinpath(@__DIR__, "PGDWarmStart.jl"))
using .PGDWarmStart

const CANDIDATE_SCHEMA_VERSION = 1
const MINIMUM_AVAILABLE_MEMORY_MB = 4_096.0

format_numeric_array(values) = join((string(Float64(value)) for value in vec(values)), ";")

function atomic_write_csv(path::String, table)
    mkpath(dirname(abspath(path)))
    temporary_path = path * ".tmp"
    CSV.write(temporary_path, table)
    mv(temporary_path, path; force = true)
    return nothing
end

function validate_existing_rows(rows::DataFrame, configuration::NamedTuple)
    nrow(rows) == 0 && return nothing
    for (column, expected) in pairs(configuration)
        column in propertynames(rows) || error("candidate cache is missing $column")
        all(value -> isequal(value, expected), rows[!, column]) ||
            error("candidate cache has a different $column configuration")
    end
    return nothing
end

function main()
    args = parse_args(ARGS)
    output_path = get(args, "out", "")
    isempty(output_path) && error("--out is required")
    sample_indices = parse_sample_spec(get(args, "samples", "1:100"))
    isempty(sample_indices) && error("--samples selects no samples")

    epsilon = parse(Float64, get(args, "epsilon", "0.1"))
    step_size = parse(Float64, get(args, "step-size", "0.01"))
    steps = parse(Int, get(args, "steps", "100"))
    restarts = parse(Int, get(args, "restarts", "20"))
    base_seed = parse(Int, get(args, "base-seed", "20260720"))
    configuration = (
        candidate_schema_version = CANDIDATE_SCHEMA_VERSION,
        epsilon = epsilon,
        step_size = step_size,
        steps = steps,
        restarts = restarts,
        base_seed = base_seed,
    )

    rows = isfile(output_path) ? CSV.read(output_path, DataFrame) : DataFrame()
    validate_existing_rows(rows, configuration)
    completed = nrow(rows) == 0 ? Set{Int}() : Set(Int.(rows.sample_index))
    pending = filter(sample -> !(sample in completed), sample_indices)
    isempty(pending) && return println("All requested PGD candidates are already cached.")

    require_available_memory(MINIMUM_AVAILABLE_MEMORY_MB)
    MIPVerify.set_log_level!(get(args, "log-level", "warn"))
    dataset = MIPVerify.read_datasets("MNIST")
    nn = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    perturbation = MIPVerify.LInfNormBoundedPerturbationFamily(epsilon)

    # Compile the batched forward and reverse passes before recording per-sample PGD time.
    warmup_input = Array(MIPVerify.get_image(dataset.test.images, first(pending)))
    warmup_true_index = MIPVerify.get_label(dataset.test.labels, first(pending)) + 1
    warmup_started = time_ns()
    projected_gradient_attack(
        nn,
        warmup_input,
        warmup_true_index;
        epsilon = epsilon,
        step_size = step_size,
        steps = min(1, steps),
        restarts = min(2, restarts),
        seed = base_seed + first(pending),
    )
    warmup_seconds = (time_ns() - warmup_started) / 1e9
    println("PGD warmup completed in $(round(warmup_seconds; digits = 3)) seconds")
    flush(stdout)

    for sample_index in pending
        available_before_mb = require_available_memory(MINIMUM_AVAILABLE_MEMORY_MB)
        input = Array(MIPVerify.get_image(dataset.test.images, sample_index))
        true_index = MIPVerify.get_label(dataset.test.labels, sample_index) + 1
        original_output = vec(input |> nn)
        predicted_index = argmax(original_output)
        sample_seed = base_seed + sample_index

        if predicted_index != true_index
            candidate = input
            verification = verify_candidate(nn, input, candidate, perturbation, true_index)
            row = merge(
                configuration,
                (
                    generated_at_utc = string(now(UTC)),
                    sample_index = sample_index,
                    true_index = true_index,
                    predicted_index = predicted_index,
                    status = "original_misclassified",
                    sample_seed = sample_seed,
                    margin = verification.margin,
                    competing_index = predicted_index,
                    best_restart = 0,
                    best_step = 0,
                    pgd_elapsed_seconds = 0.0,
                    warmup_elapsed_seconds = warmup_seconds,
                    perturbation_verified = verification.perturbation_verified,
                    target_verified = verification.target_verified,
                    verified_attack = verification.verified_attack,
                    max_abs_perturbation = 0.0,
                    restart_best_margins = "",
                    candidate_input = format_numeric_array(candidate),
                    candidate_output = format_numeric_array(verification.output),
                    original_output = format_numeric_array(original_output),
                    available_before_mb = available_before_mb,
                    available_after_mb = system_available_mb(),
                    process_rss_mb = process_rss_mb(),
                    process_peak_rss_mb = Sys.maxrss() / 2.0^20,
                ),
            )
        else
            result = projected_gradient_attack(
                nn,
                input,
                true_index;
                epsilon = epsilon,
                step_size = step_size,
                steps = steps,
                restarts = restarts,
                seed = sample_seed,
            )
            verification = verify_candidate(nn, input, result.candidate, perturbation, true_index)
            isapprox(result.margin, verification.margin; atol = 1e-10, rtol = 1e-10) ||
                error("optimized and reference PGD margins differ for sample $sample_index")
            verification.perturbation_verified ||
                error("PGD candidate is outside the perturbation set for sample $sample_index")
            status = verification.verified_attack ? "verified_pgd_attack" : "near_miss"
            row = merge(
                configuration,
                (
                    generated_at_utc = string(now(UTC)),
                    sample_index = sample_index,
                    true_index = true_index,
                    predicted_index = predicted_index,
                    status = status,
                    sample_seed = sample_seed,
                    margin = verification.margin,
                    competing_index = result.competing_index,
                    best_restart = result.restart,
                    best_step = result.step,
                    pgd_elapsed_seconds = result.elapsed_seconds,
                    warmup_elapsed_seconds = warmup_seconds,
                    perturbation_verified = verification.perturbation_verified,
                    target_verified = verification.target_verified,
                    verified_attack = verification.verified_attack,
                    max_abs_perturbation = maximum(abs.(result.candidate .- input)),
                    restart_best_margins = format_numeric_array(result.restart_best_margins),
                    candidate_input = format_numeric_array(result.candidate),
                    candidate_output = format_numeric_array(verification.output),
                    original_output = format_numeric_array(original_output),
                    available_before_mb = available_before_mb,
                    available_after_mb = system_available_mb(),
                    process_rss_mb = process_rss_mb(),
                    process_peak_rss_mb = Sys.maxrss() / 2.0^20,
                ),
            )
        end

        if nrow(rows) == 0
            rows = DataFrame([row])
        else
            push!(rows, row)
        end
        sort!(rows, :sample_index)
        atomic_write_csv(output_path, rows)
        println(
            "sample=$sample_index status=$(row.status) margin=$(round(row.margin; digits = 6)) " *
            "pgd_seconds=$(round(row.pgd_elapsed_seconds; digits = 3)) " *
            "rss_mb=$(round(row.process_rss_mb; digits = 1))",
        )
        flush(stdout)
        GC.gc()
    end
end

main()
