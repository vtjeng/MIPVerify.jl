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

function format_counts(counts::Dict{String,Int})::String
    entries = sort!(collect(counts); by = first)
    return join(("$(name)=$(count)" for (name, count) in entries), ";")
end

function sum_nonmissing_times(values)::Float64
    return safe_sum(Float64[value for value in skipmissing(values)])
end

const WK17A_RELU_LAYER_SHAPES = Tuple[(1, 14, 14, 16), (1, 7, 7, 32), (100,)]

function validate_instrumentation(stats::MIPVerify.VerificationStats)
    recorded_shapes = [layer.input_shape for layer in stats.relu_layers]
    recorded_shapes == WK17A_RELU_LAYER_SHAPES ||
        error("Expected WK17a ReLU shapes $(WK17A_RELU_LAYER_SHAPES), recorded $recorded_shapes")

    recorded_relu_total = sum(
        (
            layer.num_zero_output +
            layer.num_linear_in_input +
            layer.num_constant_output +
            layer.num_split for layer in stats.relu_layers
        );
        init = 0,
    )
    expected_relu_total = sum(prod, WK17A_RELU_LAYER_SHAPES)
    recorded_relu_total == expected_relu_total ||
        error("Expected $expected_relu_total WK17a ReLUs, recorded $recorded_relu_total")
    return nothing
end

# Single source of truth for the per-sample instrumentation columns. Samples without a
# model get all-missing fields; `collect_instrumentation_fields` asserts against this list
# so the two row shapes cannot drift apart (mismatched keys would only fail at
# `DataFrame(sample_rows)`, after the whole benchmark has run).
const INSTRUMENTATION_COLUMNS = (
    :formulation_time_seconds,
    :bound_tightening_time_seconds,
    :bound_solver_wall_time_seconds,
    :bound_solver_reported_time_seconds,
    :formulation_excluding_bound_solver_time_seconds,
    :formulation_residual_time_seconds,
    :main_solve_wall_time_seconds,
    :bound_request_count,
    :bound_solver_call_count,
    :bound_optimal_count,
    :bound_time_limit_count,
    :bound_interval_arithmetic_count,
    :bound_constant_expression_count,
    :bound_interval_cutoff_count,
    :bound_lower_skipped_count,
    :bound_simplex_iterations,
    :bound_barrier_iterations,
    :bound_node_count,
    :relu_layer_count,
    :relu_total_count,
    :relu_stable_count,
    :relu_unstable_count,
    :relu_zero_output_count,
    :relu_linear_in_input_count,
    :relu_constant_output_count,
    :num_variables,
    :num_binary_variables,
    :num_structural_constraints,
    :num_total_constraints,
    :main_node_count,
    :main_simplex_iterations,
    :main_barrier_iterations,
    :main_relative_gap,
)

missing_instrumentation_fields() =
    NamedTuple{INSTRUMENTATION_COLUMNS}(ntuple(_ -> missing, length(INSTRUMENTATION_COLUMNS)))

function collect_instrumentation_fields(d::Dict, stats::MIPVerify.VerificationStats)
    relu_bounds_time = sum((layer.bounds_time_seconds for layer in stats.relu_layers); init = 0.0)
    unscoped_bound_solver_time = sum(
        (
            group.solver_wall_time_seconds for
            ((relu_layer_index, _, _), group) in stats.bound_tightening if relu_layer_index == 0
        );
        init = 0.0,
    )
    bound_tightening_time = relu_bounds_time + unscoped_bound_solver_time
    relu_stable_count = Int(d[:ReLUStableCount])
    relu_unstable_count = Int(d[:ReLUSplitCount])
    fields = (
        formulation_time_seconds = Float64(d[:FormulationTime]),
        bound_tightening_time_seconds = bound_tightening_time,
        bound_solver_wall_time_seconds = Float64(d[:BoundSolverWallTime]),
        bound_solver_reported_time_seconds = Float64(d[:BoundSolverReportedTime]),
        formulation_excluding_bound_solver_time_seconds = Float64(
            d[:FormulationExcludingBoundSolveTime],
        ),
        formulation_residual_time_seconds = max(
            0.0,
            Float64(d[:FormulationTime]) - bound_tightening_time,
        ),
        main_solve_wall_time_seconds = Float64(d[:MainSolveWallTime]),
        bound_request_count = Int(d[:BoundRequestCount]),
        bound_solver_call_count = Int(d[:BoundSolverCallCount]),
        bound_optimal_count = Int(d[:BoundOptimalCount]),
        bound_time_limit_count = Int(d[:BoundTimeLimitCount]),
        bound_interval_arithmetic_count = Int(d[:BoundIntervalArithmeticCount]),
        bound_constant_expression_count = Int(d[:BoundConstantExpressionCount]),
        bound_interval_cutoff_count = Int(d[:BoundIntervalCutoffCount]),
        bound_lower_skipped_count = Int(d[:BoundLowerSkippedCount]),
        bound_simplex_iterations = Int(d[:BoundSimplexIterations]),
        bound_barrier_iterations = Int(d[:BoundBarrierIterations]),
        bound_node_count = Int(d[:BoundNodeCount]),
        relu_layer_count = Int(d[:ReLULayerCount]),
        relu_total_count = relu_stable_count + relu_unstable_count,
        relu_stable_count = relu_stable_count,
        relu_unstable_count = relu_unstable_count,
        relu_zero_output_count = Int(d[:ReLUZeroOutputCount]),
        relu_linear_in_input_count = Int(d[:ReLULinearInInputCount]),
        relu_constant_output_count = Int(d[:ReLUConstantOutputCount]),
        num_variables = Int(d[:NumVariables]),
        num_binary_variables = Int(d[:NumBinaryVariables]),
        num_structural_constraints = Int(d[:NumStructuralConstraints]),
        num_total_constraints = Int(d[:NumTotalConstraints]),
        main_node_count = d[:MainNodeCount],
        main_simplex_iterations = d[:MainSimplexIterations],
        main_barrier_iterations = d[:MainBarrierIterations],
        main_relative_gap = d[:MainRelativeGap],
    )
    keys(fields) == INSTRUMENTATION_COLUMNS ||
        error("Instrumentation fields do not match INSTRUMENTATION_COLUMNS")
    return fields
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
    dependency_snapshot_sha256 = dependency_snapshot_hash(dependency_snapshot)

    sample_spec = get(args, "samples", "1:100")
    sample_indices = parse_sample_spec(sample_spec)
    isempty(sample_indices) &&
        error("--samples $(sample_spec) selects no samples; nothing to benchmark")
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
    sample_rows = NamedTuple[]
    relu_layer_rows = DataFrame(
        sample_index = Int[],
        relu_layer_index = Int[],
        input_shape = String[],
        tightening_algorithm = String[],
        bounds_time_seconds = Float64[],
        constraint_time_seconds = Float64[],
        relu_total_count = Int[],
        relu_stable_count = Int[],
        relu_unstable_count = Int[],
        relu_zero_output_count = Int[],
        relu_linear_in_input_count = Int[],
        relu_constant_output_count = Int[],
    )
    tightening_rows = DataFrame(
        sample_index = Int[],
        relu_layer_index = Int[],
        tightening_algorithm = String[],
        bound_type = String[],
        request_count = Int[],
        solver_call_count = Int[],
        solver_wall_time_seconds = Float64[],
        solver_reported_time_seconds = Float64[],
        simplex_iterations = Int[],
        barrier_iterations = Int[],
        node_count = Int[],
        optimal_count = Int[],
        time_limit_count = Int[],
        interval_arithmetic_count = Int[],
        constant_expression_count = Int[],
        interval_cutoff_count = Int[],
        lower_skipped_count = Int[],
        status_counts = String[],
        skip_counts = String[],
    )
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
            collect_stats = true,
        )

        if haskey(d, :Model)
            m = d[:Model]
            status = string(d[:SolveStatus])
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
            semantic_outcome = classify_semantic_outcome(status, objective_value)
            stats = MIPVerify.get_verification_stats(m)
            stats === nothing && error("collect_stats=true returned no verification statistics")
            validate_instrumentation(stats)
            instrumentation_fields = collect_instrumentation_fields(d, stats)

            for layer in stats.relu_layers
                relu_stable_count =
                    layer.num_zero_output + layer.num_linear_in_input + layer.num_constant_output
                push!(
                    relu_layer_rows,
                    (
                        sample_index = sample_index,
                        relu_layer_index = layer.index,
                        input_shape = join(layer.input_shape, "x"),
                        tightening_algorithm = layer.tightening_algorithm,
                        bounds_time_seconds = layer.bounds_time_seconds,
                        constraint_time_seconds = layer.constraint_time_seconds,
                        relu_total_count = relu_stable_count + layer.num_split,
                        relu_stable_count = relu_stable_count,
                        relu_unstable_count = layer.num_split,
                        relu_zero_output_count = layer.num_zero_output,
                        relu_linear_in_input_count = layer.num_linear_in_input,
                        relu_constant_output_count = layer.num_constant_output,
                    ),
                )
            end

            bound_groups = sort!(collect(stats.bound_tightening); by = first)
            for ((relu_layer_index, algorithm, bound_type), group) in bound_groups
                push!(
                    tightening_rows,
                    (
                        sample_index = sample_index,
                        relu_layer_index = relu_layer_index,
                        tightening_algorithm = algorithm,
                        bound_type = bound_type,
                        request_count = group.request_count,
                        solver_call_count = group.solver_call_count,
                        solver_wall_time_seconds = group.solver_wall_time_seconds,
                        solver_reported_time_seconds = group.solver_reported_time_seconds,
                        simplex_iterations = group.simplex_iterations,
                        barrier_iterations = group.barrier_iterations,
                        node_count = group.node_count,
                        optimal_count = get(group.status_counts, MIPVerify.BOUND_STATUS_OPTIMAL, 0),
                        time_limit_count = get(
                            group.status_counts,
                            MIPVerify.BOUND_STATUS_TIME_LIMIT,
                            0,
                        ),
                        interval_arithmetic_count = get(
                            group.skip_counts,
                            MIPVerify.SKIP_INTERVAL_ARITHMETIC,
                            0,
                        ),
                        constant_expression_count = get(
                            group.skip_counts,
                            MIPVerify.SKIP_CONSTANT_EXPRESSION,
                            0,
                        ),
                        interval_cutoff_count = get(
                            group.skip_counts,
                            MIPVerify.SKIP_INTERVAL_PROVES_CUTOFF,
                            0,
                        ),
                        lower_skipped_count = get(
                            group.skip_counts,
                            MIPVerify.SKIP_LOWER_SKIPPED_BY_NONPOSITIVE_UPPER,
                            0,
                        ),
                        status_counts = format_counts(group.status_counts),
                        skip_counts = format_counts(group.skip_counts),
                    ),
                )
            end
        else
            status = "SKIPPED_PREDICTED_IN_TARGETED"
            # The unperturbed input is already misclassified, so it is a zero-distance
            # adversarial example even though model construction and solving are skipped.
            objective_value = 0.0
            objective_bound = 0.0
            semantic_outcome = classify_semantic_outcome(status, objective_value)
            instrumentation_fields = missing_instrumentation_fields()
        end
        push!(
            sample_rows,
            merge(
                (
                    sample_index = sample_index,
                    solve_status = status,
                    semantic_outcome = semantic_outcome,
                    total_time_seconds = Float64(d[:TotalTime]),
                    solve_time_seconds = haskey(d, :Model) ? Float64(d[:SolveTime]) : 0.0,
                    objective_value = objective_value,
                    objective_bound = objective_bound,
                ),
                instrumentation_fields,
            ),
        )
    end
    wall_clock_seconds = time() - wall_start
    completed_at = now()

    per_sample = DataFrame(sample_rows)
    per_sample_path = joinpath(out_dir, "benchmark_per_sample.csv")
    CSV.write(per_sample_path, per_sample)
    relu_layers_path = joinpath(out_dir, "benchmark_relu_layers.csv")
    CSV.write(relu_layers_path, relu_layer_rows)
    tightening_path = joinpath(out_dir, "benchmark_tightening.csv")
    CSV.write(tightening_path, tightening_rows)

    total_times = per_sample.total_time_seconds
    solve_times = per_sample.solve_time_seconds
    statuses = per_sample.solve_status
    semantic_outcomes = per_sample.semantic_outcome
    objective_values = per_sample.objective_value

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
        sum_formulation_time_seconds = [sum_nonmissing_times(per_sample.formulation_time_seconds)],
        sum_bound_tightening_time_seconds = [
            sum_nonmissing_times(per_sample.bound_tightening_time_seconds),
        ],
        sum_formulation_residual_time_seconds = [
            sum_nonmissing_times(per_sample.formulation_residual_time_seconds),
        ],
        sum_main_solve_wall_time_seconds = [
            sum_nonmissing_times(per_sample.main_solve_wall_time_seconds),
        ],
        median_solve_time_seconds = [median(solve_times)],
        p90_solve_time_seconds = [quantile(solve_times, 0.9)],
        num_samples = [length(sample_indices)],
        num_rows_in_summary = [nrow(per_sample)],
        num_optimal_status = [get(status_counts, "OPTIMAL", 0)],
        num_infeasible_status = [get(status_counts, "INFEASIBLE", 0)],
        num_infeasible_or_unbounded_status = [get(status_counts, "INFEASIBLE_OR_UNBOUNDED", 0)],
        num_time_limit_status = [get(status_counts, "TIME_LIMIT", 0)],
        num_skipped_predicted_in_targeted = [
            get(status_counts, "SKIPPED_PREDICTED_IN_TARGETED", 0),
        ],
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
        benchmark_schema_version = [BENCHMARK_SCHEMA_VERSION],
        semantic_outcome_schema_version = [SEMANTIC_OUTCOME_SCHEMA_VERSION],
        julia_version = [julia_version],
        dependency_snapshot_sha256 = [dependency_snapshot_sha256],
        julia_threads = [Threads.nthreads()],
        per_sample_path = [per_sample_path],
        relu_layers_path = [relu_layers_path],
        tightening_path = [tightening_path],
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
    println("  skipped_already_misclassified=$(metrics[1, :num_skipped_predicted_in_targeted])")
    println("Wrote metrics to $metrics_path")
end

main()
