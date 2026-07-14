# Skip reasons and solver statuses recorded in bound-tightening statistics. Consumers
# (e.g. the benchmark driver's CSV columns) must reference these constants rather than
# repeating the strings, so that a rename cannot silently zero their counts.
const SKIP_CONSTANT_EXPRESSION = "constant_expression"
const SKIP_INTERVAL_ARITHMETIC = "interval_arithmetic"
const SKIP_INTERVAL_PROVES_CUTOFF = "interval_proves_cutoff"
const SKIP_LOWER_SKIPPED_BY_NONPOSITIVE_UPPER = "lower_skipped_by_nonpositive_upper"
const SKIP_UPPER_SKIPPED_BY_NONNEGATIVE_INTERVAL_LOWER = "upper_skipped_by_nonnegative_interval_lower"
const BOUND_STATUS_OPTIMAL = string(MathOptInterface.OPTIMAL)
const BOUND_STATUS_TIME_LIMIT = string(MathOptInterface.TIME_LIMIT)

mutable struct BoundTighteningStats
    request_count::Int
    solver_call_count::Int
    solver_wall_time_seconds::Float64
    solver_reported_time_seconds::Float64
    simplex_iterations::Int
    barrier_iterations::Int
    node_count::Int
    status_counts::Dict{String,Int}
    skip_counts::Dict{String,Int}
end

function BoundTighteningStats()
    return BoundTighteningStats(0, 0, 0.0, 0.0, 0, 0, 0, Dict{String,Int}(), Dict{String,Int}())
end

mutable struct ReLULayerStats
    index::Int
    input_shape::Tuple
    tightening_algorithm::String
    bounds_time_seconds::Float64
    constraint_time_seconds::Float64
    num_zero_output::Int
    num_linear_in_input::Int
    num_constant_output::Int
    num_split::Int
end

function ReLULayerStats(index::Int, input_shape::Tuple, tightening_algorithm::String)
    return ReLULayerStats(index, input_shape, tightening_algorithm, 0.0, 0.0, 0, 0, 0, 0)
end

mutable struct VerificationStats
    current_relu_layer_index::Int
    bound_tightening::Dict{Tuple{Int,String,String},BoundTighteningStats}
    relu_layers::Vector{ReLULayerStats}
end

VerificationStats() =
    VerificationStats(0, Dict{Tuple{Int,String,String},BoundTighteningStats}(), ReLULayerStats[])

const VERIFICATION_STATS_TASK_LOCAL_KEY = :MIPVerifyVerificationStats

"""
    with_verification_stats(f, stats)

Run `f` with `stats` as the task-local fallback for JuMP expressions that have
no owner model. Passing `nothing` disables the fallback within this scope. The
previous task-local value is restored when `f` returns or throws.
"""
function with_verification_stats(f::Function, stats::Union{Nothing,VerificationStats})
    return task_local_storage(f, VERIFICATION_STATS_TASK_LOCAL_KEY, stats)
end

struct MIPVerifyExt
    tightening_algorithm::TighteningAlgorithm
    stats::Union{Nothing,VerificationStats}
end

# Preserve the pre-instrumentation constructor; omitting stats disables collection.
MIPVerifyExt(tightening_algorithm::TighteningAlgorithm) =
    MIPVerifyExt(tightening_algorithm, nothing)

"""
    get_verification_stats(model)
    get_verification_stats(x)
    get_verification_stats(xs)

Return the verification statistics associated with the input, or `nothing` when
none are available.

Variables and affine expressions use statistics attached to their owner model.
Arrays use the first element with an owner model. If an expression has no owner
model, or every array element lacks one, use the current task-local
`with_verification_stats` scope. This fallback lets constant expressions
participate in the surrounding model's statistics collection.
"""
function get_verification_stats(model::Model)::Union{Nothing,VerificationStats}
    extension = get(model.ext, :MIPVerify, nothing)
    if extension isa MIPVerifyExt
        return extension.stats
    end
    return nothing
end

function scoped_verification_stats()::Union{Nothing,VerificationStats}
    scoped_stats = get(task_local_storage(), VERIFICATION_STATS_TASK_LOCAL_KEY, nothing)
    return scoped_stats isa VerificationStats ? scoped_stats : nothing
end

function get_verification_stats(
    x::Union{JuMP.VariableRef,JuMP.GenericAffExpr},
)::Union{Nothing,VerificationStats}
    model = owner_model(x)
    return model === nothing ? scoped_verification_stats() : get_verification_stats(model)
end

function get_verification_stats(
    xs::AbstractArray{T},
)::Union{Nothing,VerificationStats} where {T<:Union{JuMP.VariableRef,JuMP.GenericAffExpr}}
    model = owner_model(xs)
    return model === nothing ? scoped_verification_stats() : get_verification_stats(model)
end

elapsed_seconds(start_time_ns::UInt64)::Float64 = (time_ns() - start_time_ns) / 1.0e9

"""
    get_bound_tightening_stats(stats, tightening_algorithm, bound_type)

Return the mutable statistics group for the current ReLU layer, tightening
algorithm, and bound type. Create and store an empty group if none exists. Layer
index `0` records work outside a ReLU layer.
"""
function get_bound_tightening_stats(
    stats::VerificationStats,
    tightening_algorithm::TighteningAlgorithm,
    bound_type::String,
)::BoundTighteningStats
    key = (stats.current_relu_layer_index, string(tightening_algorithm), bound_type)
    return get!(stats.bound_tightening, key) do
        BoundTighteningStats()
    end
end

"""
    record_bound_skip!(stats, tightening_algorithm, bound_type, reason)

Record one logical bound request as skipped without recording a solver call.
Increment the request count and the skip count for `reason` in the current
layer, algorithm, and bound-type group. If `stats` is `nothing`, do nothing.

Do not also call `record_bound_request!` for the same request.
"""
function record_bound_skip!(
    stats::Union{Nothing,VerificationStats},
    tightening_algorithm::TighteningAlgorithm,
    bound_type::String,
    reason::String,
)::Nothing
    stats === nothing && return nothing
    aggregate = get_bound_tightening_stats(stats, tightening_algorithm, bound_type)
    aggregate.request_count += 1
    aggregate.skip_counts[reason] = get(aggregate.skip_counts, reason, 0) + 1
    return nothing
end

"""
    record_bound_request!(stats, tightening_algorithm, bound_type)

Count a bound-tightening request after all skip conditions have been ruled out.
`record_bound_skip!` counts skipped requests and their reasons. A later
`record_bound_solve!` records the optimizer call without incrementing the request
count. If `stats` is `nothing`, do nothing.
"""
function record_bound_request!(
    stats::Union{Nothing,VerificationStats},
    tightening_algorithm::TighteningAlgorithm,
    bound_type::String,
)::Nothing
    stats === nothing && return nothing
    aggregate = get_bound_tightening_stats(stats, tightening_algorithm, bound_type)
    aggregate.request_count += 1
    return nothing
end

"""
    safe_solve_metric(f, model, default)

Return `f(model)` when it is a finite, nonnegative `Real`. Return `default` when
`f` throws `MathOptInterface.UnsupportedError` or
`MathOptInterface.NotAllowedError`, or when it returns any other value. Rethrow
all other errors.
"""
function safe_solve_metric(f, model::Model, default)
    value = try
        f(model)
    catch error
        # Only attribute-availability errors mean "this solver cannot report the metric";
        # anything else is a genuine bug and must propagate.
        if !(error isa Union{MathOptInterface.NotAllowedError,MathOptInterface.UnsupportedError})
            rethrow()
        end
        return default
    end
    if !(value isa Real) || !isfinite(value) || value < 0
        return default
    end
    return value
end

"""
    record_bound_solve!(stats, tightening_algorithm, bound_type, model, status,
                        wall_time_seconds)

Record one completed bound-tightening solver call in the current layer,
algorithm, and bound-type group. Add the measured wall time, termination status,
and available nonnegative solver timing and work metrics. Unsupported or invalid
metrics contribute zero.

This does not increment the logical request count, which callers record before
solving. If `stats` is `nothing`, do nothing.
"""
function record_bound_solve!(
    stats::Union{Nothing,VerificationStats},
    tightening_algorithm::TighteningAlgorithm,
    bound_type::String,
    model::Model,
    status,
    wall_time_seconds::Float64,
)::Nothing
    stats === nothing && return nothing
    aggregate = get_bound_tightening_stats(stats, tightening_algorithm, bound_type)
    aggregate.solver_call_count += 1
    aggregate.solver_wall_time_seconds += wall_time_seconds
    aggregate.solver_reported_time_seconds += safe_solve_metric(JuMP.solve_time, model, 0.0)
    aggregate.simplex_iterations += safe_solve_metric(JuMP.simplex_iterations, model, 0)
    aggregate.barrier_iterations += safe_solve_metric(JuMP.barrier_iterations, model, 0)
    aggregate.node_count += safe_solve_metric(JuMP.node_count, model, 0)
    status_name = string(status)
    aggregate.status_counts[status_name] = get(aggregate.status_counts, status_name, 0) + 1
    return nothing
end

"""
    begin_relu_layer!(stats, input_shape, tightening_algorithm)

Append a ReLU-layer record and make its one-based index current so subsequent
bound-tightening statistics are attributed to that layer. `finish_relu_layer!`
resets the current index to `0`, which identifies work outside a ReLU layer.
Return the new record, or `nothing` when statistics collection is disabled.
"""
function begin_relu_layer!(
    stats::Union{Nothing,VerificationStats},
    input_shape::Tuple,
    tightening_algorithm::TighteningAlgorithm,
)::Union{Nothing,ReLULayerStats}
    stats === nothing && return nothing
    index = length(stats.relu_layers) + 1
    layer = ReLULayerStats(index, input_shape, string(tightening_algorithm))
    push!(stats.relu_layers, layer)
    stats.current_relu_layer_index = index
    return layer
end

"""
    finish_relu_layer!(stats, layer, bounds_time_seconds, constraint_time_seconds,
                       num_zero_output, num_linear_in_input, num_constant_output,
                       num_split)

Store the completed ReLU layer's timings and phase counts in `layer`. Reset the
current layer index to `0` so later bound-tightening work is recorded as outside
a ReLU layer. If `stats` or `layer` is `nothing`, leave the statistics unchanged.
"""
function finish_relu_layer!(
    stats::Union{Nothing,VerificationStats},
    layer::Union{Nothing,ReLULayerStats},
    bounds_time_seconds::Float64,
    constraint_time_seconds::Float64,
    num_zero_output::Int,
    num_linear_in_input::Int,
    num_constant_output::Int,
    num_split::Int,
)::Nothing
    if stats === nothing || layer === nothing
        return nothing
    end
    layer.bounds_time_seconds = bounds_time_seconds
    layer.constraint_time_seconds = constraint_time_seconds
    layer.num_zero_output = num_zero_output
    layer.num_linear_in_input = num_linear_in_input
    layer.num_constant_output = num_constant_output
    layer.num_split = num_split
    stats.current_relu_layer_index = 0
    return nothing
end

function count_status(stats::VerificationStats, status::String)::Int
    return sum(
        (get(group.status_counts, status, 0) for group in values(stats.bound_tightening));
        init = 0,
    )
end

function count_skip(stats::VerificationStats, reason::String)::Int
    return sum(
        (get(group.skip_counts, reason, 0) for group in values(stats.bound_tightening));
        init = 0,
    )
end

"""
    summarize_verification_stats(stats)
    summarize_verification_stats(model)

Aggregate bound-tightening and ReLU-layer records into the flat statistics fields
returned by `find_adversarial_example(...; collect_stats=true)`. Formulation
timing, model-size, and main-solver fields are added separately.

The model overload returns an empty dictionary when the model has no verification
statistics.
"""
function summarize_verification_stats(stats::VerificationStats)::Dict{Symbol,Any}
    bound_groups = values(stats.bound_tightening)
    relu_layers = stats.relu_layers
    num_zero_output = sum((layer.num_zero_output for layer in relu_layers); init = 0)
    num_linear_in_input = sum((layer.num_linear_in_input for layer in relu_layers); init = 0)
    num_constant_output = sum((layer.num_constant_output for layer in relu_layers); init = 0)
    num_split = sum((layer.num_split for layer in relu_layers); init = 0)
    return Dict(
        :BoundRequestCount => sum((group.request_count for group in bound_groups); init = 0),
        :BoundSolverCallCount => sum((group.solver_call_count for group in bound_groups); init = 0),
        :BoundSolverWallTime =>
            sum((group.solver_wall_time_seconds for group in bound_groups); init = 0.0),
        :BoundSolverReportedTime =>
            sum((group.solver_reported_time_seconds for group in bound_groups); init = 0.0),
        :BoundSimplexIterations =>
            sum((group.simplex_iterations for group in bound_groups); init = 0),
        :BoundBarrierIterations =>
            sum((group.barrier_iterations for group in bound_groups); init = 0),
        :BoundNodeCount => sum((group.node_count for group in bound_groups); init = 0),
        :BoundOptimalCount => count_status(stats, BOUND_STATUS_OPTIMAL),
        :BoundTimeLimitCount => count_status(stats, BOUND_STATUS_TIME_LIMIT),
        :BoundIntervalArithmeticCount => count_skip(stats, SKIP_INTERVAL_ARITHMETIC),
        :BoundConstantExpressionCount => count_skip(stats, SKIP_CONSTANT_EXPRESSION),
        :BoundIntervalCutoffCount => count_skip(stats, SKIP_INTERVAL_PROVES_CUTOFF),
        :BoundLowerSkippedCount => count_skip(stats, SKIP_LOWER_SKIPPED_BY_NONPOSITIVE_UPPER),
        :ReLULayerCount => length(relu_layers),
        :ReLUZeroOutputCount => num_zero_output,
        :ReLULinearInInputCount => num_linear_in_input,
        :ReLUConstantOutputCount => num_constant_output,
        :ReLUSplitCount => num_split,
        :ReLUStableCount => num_zero_output + num_linear_in_input + num_constant_output,
    )
end

function summarize_verification_stats(model::Model)::Dict{Symbol,Any}
    stats = get_verification_stats(model)
    return stats === nothing ? Dict{Symbol,Any}() : summarize_verification_stats(stats)
end
