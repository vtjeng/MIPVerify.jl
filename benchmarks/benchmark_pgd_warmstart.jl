using CSV
using DataFrames
using Dates
using HiGHS
using MIPVerify
using Pkg
using SHA

include(joinpath(@__DIR__, "BenchmarkHelpers.jl"))
using .BenchmarkHelpers
include(joinpath(@__DIR__, "PGDWarmStart.jl"))
using .PGDWarmStart

const WARMSTART_BENCHMARK_SCHEMA_VERSION = 1
const MINIMUM_AVAILABLE_MEMORY_MB = 4_096.0

format_numeric_array(values) = join((string(Float64(value)) for value in vec(values)), ";")

function atomic_write_csv(path::String, table)
    mkpath(dirname(abspath(path)))
    temporary_path = path * ".tmp"
    CSV.write(temporary_path, table)
    mv(temporary_path, path; force = true)
    return nothing
end

function append_row!(table::DataFrame, row::NamedTuple, path::String)
    if nrow(table) == 0 && ncol(table) == 0
        append!(table, DataFrame([row]); cols = :union)
    else
        push!(table, row; cols = :union, promote = true)
    end
    atomic_write_csv(path, table)
    return nothing
end

function append_rows!(table::DataFrame, rows::Vector{<:NamedTuple}, path::String)
    isempty(rows) && return nothing
    incoming = DataFrame(rows)
    if nrow(table) == 0 && ncol(table) == 0
        append!(table, incoming; cols = :union)
    else
        append!(table, incoming; cols = :union, promote = true)
    end
    atomic_write_csv(path, table)
    return nothing
end

function parse_numeric_array(serialized, shape)
    values = parse.(Float64, split(string(serialized), ";"))
    length(values) == prod(shape) || error("serialized candidate has the wrong size")
    return reshape(values, shape)
end

function parse_tightening_algorithm(name::String)::MIPVerify.TighteningAlgorithm
    normalized = lowercase(strip(name))
    normalized == "interval_arithmetic" && return MIPVerify.interval_arithmetic
    normalized == "lp" && return MIPVerify.lp
    normalized == "mip" && return MIPVerify.mip
    error("unsupported tightening algorithm: $name")
end

function load_cohort(args)::DataFrame
    if haskey(args, "cohort-file")
        cohort = CSV.read(args["cohort-file"], DataFrame)
        required = [:sample_index, :stratum]
        all(column -> column in propertynames(cohort), required) ||
            error("--cohort-file must contain sample_index and stratum")
        cohort = select(cohort, required)
    else
        samples = parse_sample_spec(get(args, "samples", "1:10"))
        stratum = get(args, "stratum", "unspecified")
        cohort = DataFrame(sample_index = samples, stratum = fill(stratum, length(samples)))
    end
    nrow(unique(cohort, :sample_index)) == nrow(cohort) ||
        error("cohort contains duplicate sample indices")
    sort!(cohort, :sample_index)
    return cohort
end

function validate_candidate_configuration(candidates::DataFrame, configuration::NamedTuple)
    for (column, expected) in pairs(configuration)
        column in propertynames(candidates) || error("candidate cache is missing $column")
        all(value -> isequal(value, expected), candidates[!, column]) ||
            error("candidate cache has a different $column configuration")
    end
    return nothing
end

function write_or_validate_run_configuration(path::String, configuration::NamedTuple)
    if isfile(path)
        recorded = CSV.read(path, DataFrame)
        nrow(recorded) == 1 || error("run configuration must have exactly one row")
        for (column, expected) in pairs(configuration)
            column in propertynames(recorded) || error("run configuration is missing $column")
            isequal(recorded[1, column], expected) || error("run configuration differs for $column")
        end
    else
        atomic_write_csv(path, DataFrame([configuration]))
    end
    return nothing
end

function log_start_observation(path::String)
    isfile(path) || return (present = false, accepted = missing, rejected = missing, excerpt = "")
    lines = filter(line -> occursin("mip start", lowercase(line)), readlines(path))
    isempty(lines) && return (present = false, accepted = missing, rejected = missing, excerpt = "")
    lowered = lowercase.(lines)
    rejected = any(
        line -> occursin("infeasible", line) || occursin("cannot yield feasible", line),
        lowered,
    )
    accepted = any(
        line ->
            occursin("mip start solution is feasible", line) ||
                occursin("mip start provided solution", line),
        lowered,
    )
    return (
        present = true,
        accepted = accepted && !rejected,
        rejected = rejected,
        excerpt = join(lines, " | "),
    )
end

function optional_verification_field(verification, field::Symbol)
    isnothing(verification) && return missing
    return getproperty(verification, field)
end

function main()
    args = parse_args(ARGS)
    output_dir = get(args, "out", "")
    isempty(output_dir) && error("--out is required")
    candidate_path = get(args, "candidates", "")
    isempty(candidate_path) && error("--candidates is required")
    isfile(candidate_path) || error("candidate cache does not exist: $candidate_path")
    mkpath(output_dir)

    cohort = load_cohort(args)
    block_ids = parse_sample_spec(get(args, "blocks", "1:3"))
    all(block_id -> block_id > 0, block_ids) || error("block identifiers must be positive")
    epsilon = parse(Float64, get(args, "epsilon", "0.1"))
    step_size = parse(Float64, get(args, "step-size", "0.01"))
    steps = parse(Int, get(args, "steps", "100"))
    restarts = parse(Int, get(args, "restarts", "20"))
    base_seed = parse(Int, get(args, "base-seed", "20260720"))
    tightening_name = lowercase(get(args, "tightening", "lp"))
    tightening_algorithm = parse_tightening_algorithm(tightening_name)
    tightening_time_limit = parse(Float64, get(args, "tightening-time-limit", "20"))
    main_time_limit = parse(Float64, get(args, "main-time-limit", "30"))
    full_start_time_limit = parse(Float64, get(args, "full-start-time-limit", "30"))
    robust_tolerance = parse(Float64, get(args, "robust-tolerance", "1e-8"))

    candidates = CSV.read(candidate_path, DataFrame)
    validate_candidate_configuration(
        candidates,
        (
            candidate_schema_version = 1,
            epsilon = epsilon,
            step_size = step_size,
            steps = steps,
            restarts = restarts,
            base_seed = base_seed,
        ),
    )
    candidate_rows = Dict(Int(row.sample_index) => row for row in eachrow(candidates))
    all(sample -> haskey(candidate_rows, sample), cohort.sample_index) ||
        error("candidate cache does not cover the cohort")

    cohort_descriptor = join(("$(row.sample_index):$(row.stratum)" for row in eachrow(cohort)), ";")
    configuration = (
        warmstart_benchmark_schema_version = WARMSTART_BENCHMARK_SCHEMA_VERSION,
        cohort = cohort_descriptor,
        blocks = join(block_ids, ";"),
        epsilon = epsilon,
        step_size = step_size,
        steps = steps,
        restarts = restarts,
        base_seed = base_seed,
        tightening = tightening_name,
        tightening_time_limit_seconds = tightening_time_limit,
        main_time_limit_seconds = main_time_limit,
        full_start_time_limit_seconds = full_start_time_limit,
        robust_tolerance = robust_tolerance,
        highs_threads = 1,
        highs_parallel = "off",
        highs_random_seed = 0,
        candidate_cache_sha256 = bytes2hex(sha256(read(candidate_path))),
        julia_version = string(VERSION),
        commit_sha = readchomp(`git rev-parse HEAD`),
    )
    write_or_validate_run_configuration(
        joinpath(output_dir, "run_configuration.csv"),
        configuration,
    )
    atomic_write_csv(joinpath(output_dir, "cohort.csv"), cohort)

    dependency_path = joinpath(output_dir, "dependency_versions.csv")
    if !isfile(dependency_path)
        write_dependency_snapshot(dependency_path, collect_dependency_snapshot())
        cp(active_manifest_path(), joinpath(output_dir, "dependency_manifest.toml"); force = true)
    end

    treatment_path = joinpath(output_dir, "benchmark_per_treatment.csv")
    sample_path = joinpath(output_dir, "benchmark_samples.csv")
    trace_path = joinpath(output_dir, "benchmark_trace.csv")
    treatment_rows = isfile(treatment_path) ? CSV.read(treatment_path, DataFrame) : DataFrame()
    sample_rows = isfile(sample_path) ? CSV.read(sample_path, DataFrame) : DataFrame()
    trace_rows = isfile(trace_path) ? CSV.read(trace_path, DataFrame) : DataFrame()
    completed_treatments =
        nrow(treatment_rows) == 0 ? Set{Tuple{Int,Int,String}}() :
        Set(
            (Int(row.sample_index), Int(row.block_id), string(row.treatment)) for
            row in eachrow(treatment_rows)
        )
    completed_samples = nrow(sample_rows) == 0 ? Set{Int}() : Set(Int.(sample_rows.sample_index))

    require_available_memory(MINIMUM_AVAILABLE_MEMORY_MB)
    MIPVerify.set_log_level!(get(args, "log-level", "warn"))
    dataset = MIPVerify.read_datasets("MNIST")
    nn = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    perturbation = MIPVerify.LInfNormBoundedPerturbationFamily(epsilon)
    tightening_options = Dict(
        "output_flag" => false,
        "time_limit" => tightening_time_limit,
        "threads" => 1,
        "parallel" => "off",
        "random_seed" => 0,
    )

    for cohort_row in eachrow(cohort)
        sample_index = Int(cohort_row.sample_index)
        stratum = string(cohort_row.stratum)
        candidate_row = candidate_rows[sample_index]
        pending_keys = [
            (sample_index, block_id, string(variant.name)) for block_id in block_ids for
            variant in ordered_variants(block_id) if
            !((sample_index, block_id, string(variant.name)) in completed_treatments)
        ]
        if isempty(pending_keys) && sample_index in completed_samples
            println("sample=$sample_index already complete")
            continue
        end

        input = Array(MIPVerify.get_image(dataset.test.images, sample_index))
        true_index = MIPVerify.get_label(dataset.test.labels, sample_index) + 1
        exact_candidate = parse_numeric_array(candidate_row.candidate_input, size(input))
        exact_verification = verify_candidate(nn, input, exact_candidate, perturbation, true_index)
        isapprox(
            exact_verification.margin,
            Float64(candidate_row.margin);
            atol = 1e-10,
            rtol = 1e-10,
        ) || error("cached candidate margin changed for sample $sample_index")
        solver_candidate = inward_project_linf(exact_candidate, input, epsilon)
        solver_verification =
            verify_candidate(nn, input, solver_candidate, perturbation, true_index)

        if exact_verification.verified_attack ||
           string(candidate_row.status) == "original_misclassified"
            if !(sample_index in completed_samples)
                append_row!(
                    sample_rows,
                    (
                        warmstart_benchmark_schema_version = WARMSTART_BENCHMARK_SCHEMA_VERSION,
                        sample_index = sample_index,
                        stratum = stratum,
                        sample_status = exact_verification.verified_attack ? "verified_pgd_attack" :
                                        "original_misclassified",
                        true_index = true_index,
                        pgd_margin = exact_verification.margin,
                        solver_start_margin = solver_verification.margin,
                        pgd_elapsed_seconds = Float64(candidate_row.pgd_elapsed_seconds),
                        formulation_time_seconds = missing,
                        full_start_completion_status = missing,
                        full_start_completion_time_seconds = missing,
                        num_variables = missing,
                        num_binary_variables = missing,
                        num_structural_constraints = missing,
                        variable_bounds_sha256 = missing,
                        rss_before_formulation_mb = process_rss_mb(),
                        rss_after_formulation_mb = missing,
                        rss_after_completion_mb = missing,
                        process_peak_rss_mb = Sys.maxrss() / 2.0^20,
                        system_available_mb = system_available_mb(),
                    ),
                    sample_path,
                )
                push!(completed_samples, sample_index)
            end
            println("sample=$sample_index skipped: verified candidate attack")
            flush(stdout)
            continue
        end

        require_available_memory(MINIMUM_AVAILABLE_MEMORY_MB)
        GC.gc()
        rss_before_formulation = process_rss_mb()
        base = build_worst_margin_problem(
            nn,
            input,
            true_index,
            HiGHS.Optimizer,
            tightening_options;
            pp = perturbation,
            tightening_algorithm = tightening_algorithm,
        )
        rss_after_formulation = process_rss_mb()

        full_start = nothing
        completion_status = "not_attempted"
        completion_time = missing
        completion_error = ""
        try
            full_start =
                complete_full_start(base, solver_candidate; time_limit = full_start_time_limit)
            completion_status = string(full_start.termination_status)
            completion_time = full_start.completion_time_seconds
        catch error_value
            completion_status = "failed"
            completion_error = sprint(showerror, error_value)
        end
        GC.gc()
        rss_after_completion = process_rss_mb()

        if !(sample_index in completed_samples)
            stats = base.formulation_stats
            append_row!(
                sample_rows,
                (
                    warmstart_benchmark_schema_version = WARMSTART_BENCHMARK_SCHEMA_VERSION,
                    sample_index = sample_index,
                    stratum = stratum,
                    sample_status = "verification_ready",
                    true_index = true_index,
                    pgd_margin = exact_verification.margin,
                    solver_start_margin = solver_verification.margin,
                    pgd_elapsed_seconds = Float64(candidate_row.pgd_elapsed_seconds),
                    formulation_time_seconds = base.formulation_time_seconds,
                    full_start_completion_status = completion_status,
                    full_start_completion_time_seconds = completion_time,
                    full_start_completion_error = completion_error,
                    num_variables = base.signature.num_variables,
                    num_binary_variables = base.signature.num_binary_variables,
                    num_structural_constraints = base.signature.num_structural_constraints,
                    variable_bounds_sha256 = base.signature.variable_bounds_sha256,
                    relu_split_count = get(stats, :ReLUSplitCount, missing),
                    relu_stable_count = get(stats, :ReLUStableCount, missing),
                    bound_solver_wall_time_seconds = get(stats, :BoundSolverWallTime, missing),
                    bound_simplex_iterations = get(stats, :BoundSimplexIterations, missing),
                    rss_before_formulation_mb = rss_before_formulation,
                    rss_after_formulation_mb = rss_after_formulation,
                    rss_after_completion_mb = rss_after_completion,
                    process_peak_rss_mb = Sys.maxrss() / 2.0^20,
                    system_available_mb = system_available_mb(),
                ),
                sample_path,
            )
            push!(completed_samples, sample_index)
        end

        for block_id in block_ids
            variants = ordered_variants(block_id)
            for (treatment_order, variant) in enumerate(variants)
                key = (sample_index, block_id, string(variant.name))
                key in completed_treatments && continue
                require_available_memory(MINIMUM_AVAILABLE_MEMORY_MB)

                if variant.name == :pgd_full && isnothing(full_start)
                    append_row!(
                        treatment_rows,
                        (
                            warmstart_benchmark_schema_version = WARMSTART_BENCHMARK_SCHEMA_VERSION,
                            sample_index = sample_index,
                            stratum = stratum,
                            block_id = block_id,
                            treatment_order = treatment_order,
                            treatment = string(variant.name),
                            start_source = string(variant.source),
                            start_coverage = string(variant.coverage),
                            outcome = "full_start_completion_failed",
                            completion_error = completion_error,
                        ),
                        treatment_path,
                    )
                    push!(completed_treatments, key)
                    continue
                end

                available_before_copy = system_available_mb()
                rss_before_copy = process_rss_mb()
                copy_started = time_ns()
                treatment_problem = copy_problem(base)
                copy_time = (time_ns() - copy_started) / 1e9
                rss_after_copy = process_rss_mb()
                apply_started = time_ns()
                apply_variant_start!(
                    treatment_problem,
                    variant,
                    input,
                    solver_candidate;
                    full_start = full_start,
                )
                apply_time = (time_ns() - apply_started) / 1e9
                start_margin = if variant.name == :cold
                    missing
                elseif variant.name == :original_sparse
                    worst_margin(vec(input |> nn), true_index)
                else
                    solver_verification.margin
                end
                log_path = joinpath(
                    output_dir,
                    "logs",
                    "sample-$(sample_index)-block-$(block_id)-$(variant.name).log",
                )
                result = solve_worst_margin!(
                    treatment_problem,
                    nn,
                    input,
                    perturbation;
                    time_limit = main_time_limit,
                    robust_tolerance = robust_tolerance,
                    log_path = log_path,
                )
                start_log = log_start_observation(log_path)
                observed_start_incumbent = if ismissing(start_margin)
                    missing
                else
                    any(
                        trace_row ->
                            isfinite(trace_row.primal_bound) &&
                                trace_row.primal_bound >= start_margin - 1e-7,
                        result.trace.rows,
                    )
                end
                start_used = if ismissing(start_margin)
                    missing
                elseif start_log.accepted === true || observed_start_incumbent === true
                    true
                elseif start_log.rejected === true
                    false
                else
                    missing
                end
                charged_pgd_time =
                    variant.source == :pgd ? Float64(candidate_row.pgd_elapsed_seconds) : 0.0
                charged_completion_time = variant.name == :pgd_full ? Float64(completion_time) : 0.0
                end_to_end_time =
                    base.formulation_time_seconds +
                    copy_time +
                    apply_time +
                    result.wall_time_seconds +
                    charged_pgd_time +
                    charged_completion_time

                append_rows!(
                    trace_rows,
                    [
                        merge(
                            (
                                sample_index = sample_index,
                                stratum = stratum,
                                block_id = block_id,
                                treatment_order = treatment_order,
                                treatment = string(variant.name),
                            ),
                            trace_row,
                        ) for trace_row in result.trace.rows
                    ],
                    trace_path,
                )

                append_row!(
                    treatment_rows,
                    (
                        warmstart_benchmark_schema_version = WARMSTART_BENCHMARK_SCHEMA_VERSION,
                        sample_index = sample_index,
                        stratum = stratum,
                        block_id = block_id,
                        treatment_order = treatment_order,
                        treatment = string(variant.name),
                        start_source = string(variant.source),
                        start_coverage = string(variant.coverage),
                        pgd_margin = exact_verification.margin,
                        solver_start_margin = solver_verification.margin,
                        applied_start_margin = start_margin,
                        mip_start_log_present = start_log.present,
                        mip_start_log_accepted = start_log.accepted,
                        mip_start_log_rejected = start_log.rejected,
                        observed_start_incumbent = observed_start_incumbent,
                        mip_start_used = start_used,
                        mip_start_log_excerpt = start_log.excerpt,
                        outcome = result.outcome,
                        termination_status = result.termination_status,
                        primal_status = result.primal_status,
                        objective_value = result.objective_value,
                        objective_bound = result.objective_bound,
                        stopped_on_negative_bound = result.stopped_on_negative_bound,
                        witness_margin = optional_verification_field(result.verification, :margin),
                        witness_target_verified = optional_verification_field(
                            result.verification,
                            :target_verified,
                        ),
                        witness_perturbation_verified = optional_verification_field(
                            result.verification,
                            :perturbation_verified,
                        ),
                        witness_verified_attack = optional_verification_field(
                            result.verification,
                            :verified_attack,
                        ),
                        witness_input = isnothing(result.candidate) ? missing :
                                        format_numeric_array(result.candidate),
                        witness_output = isnothing(result.verification) ? missing :
                                         format_numeric_array(result.verification.output),
                        formulation_time_seconds = base.formulation_time_seconds,
                        model_copy_time_seconds = copy_time,
                        apply_start_time_seconds = apply_time,
                        main_solve_wall_time_seconds = result.wall_time_seconds,
                        solver_time_seconds = result.solver_time_seconds,
                        charged_pgd_time_seconds = charged_pgd_time,
                        charged_completion_time_seconds = charged_completion_time,
                        end_to_end_time_seconds = end_to_end_time,
                        node_count = result.node_count,
                        simplex_iterations = result.simplex_iterations,
                        relative_gap = result.relative_gap,
                        rss_before_copy_mb = rss_before_copy,
                        rss_after_copy_mb = rss_after_copy,
                        rss_before_solve_mb = result.rss_before_mb,
                        rss_after_solve_mb = result.rss_after_mb,
                        process_peak_rss_mb = result.max_rss_mb,
                        system_available_before_copy_mb = available_before_copy,
                        system_available_after_solve_mb = system_available_mb(),
                        num_variables = base.signature.num_variables,
                        num_binary_variables = base.signature.num_binary_variables,
                        num_structural_constraints = base.signature.num_structural_constraints,
                        variable_bounds_sha256 = base.signature.variable_bounds_sha256,
                    ),
                    treatment_path,
                )
                push!(completed_treatments, key)
                println(
                    "sample=$sample_index block=$block_id treatment=$(variant.name) " *
                    "outcome=$(result.outcome) solve=$(round(result.wall_time_seconds; digits = 3)) " *
                    "nodes=$(result.node_count) simplex=$(result.simplex_iterations) " *
                    "rss_mb=$(round(result.rss_after_mb; digits = 1))",
                )
                flush(stdout)
                result = nothing
                treatment_problem = nothing
                GC.gc()
            end
        end
        full_start = nothing
        base = nothing
        GC.gc()
    end
end

main()
