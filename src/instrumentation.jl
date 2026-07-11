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
    formulation_time_seconds::Float64
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

function with_verification_stats(f::Function, stats::Union{Nothing,VerificationStats})
    return task_local_storage(f, VERIFICATION_STATS_TASK_LOCAL_KEY, stats)
end

struct MIPVerifyExt
    tightening_algorithm::TighteningAlgorithm
    stats::Union{Nothing,VerificationStats}
end

MIPVerifyExt(tightening_algorithm::TighteningAlgorithm) =
    MIPVerifyExt(tightening_algorithm, nothing)

function MIPVerifyExt(tightening_algorithm::TighteningAlgorithm, collect_stats::Bool)
    stats = collect_stats ? VerificationStats() : nothing
    return MIPVerifyExt(tightening_algorithm, stats)
end

function get_verification_stats(model::Model)::Union{Nothing,VerificationStats}
    extension = get(model.ext, :MIPVerify, nothing)
    if extension isa MIPVerifyExt
        return extension.stats
    end
    return nothing
end

function get_verification_stats(
    xs::AbstractArray{T},
)::Union{Nothing,VerificationStats} where {T<:Union{JuMP.VariableRef,JuMP.GenericAffExpr}}
    for x in xs
        model = owner_model(x)
        if model !== nothing
            return get_verification_stats(model)
        end
    end
    scoped_stats = get(task_local_storage(), VERIFICATION_STATS_TASK_LOCAL_KEY, nothing)
    return scoped_stats isa VerificationStats ? scoped_stats : nothing
end

elapsed_seconds(start_time_ns::UInt64)::Float64 = (time_ns() - start_time_ns) / 1.0e9

function num_model_constraints(model::Model; count_variable_in_set_constraints::Bool)::Int
    total = 0
    for (function_type, set_type) in JuMP.list_of_constraint_types(model)
        if count_variable_in_set_constraints || function_type != JuMP.VariableRef
            total += JuMP.num_constraints(model, function_type, set_type)
        end
    end
    return total
end

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

function safe_solve_metric(f, model::Model, default)
    value = try
        f(model)
    catch error
        if error isa InterruptException || error isa OutOfMemoryError || error isa StackOverflowError
            rethrow()
        end
        return default
    end
    if !(value isa Real) || !isfinite(value) || value < 0
        return default
    end
    return value
end

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

function finish_relu_layer!(
    stats::Union{Nothing,VerificationStats},
    layer::Union{Nothing,ReLULayerStats},
    bounds_time_seconds::Float64,
    formulation_time_seconds::Float64,
    num_zero_output::Int,
    num_linear_in_input::Int,
    num_constant_output::Int,
    num_split::Int,
)::Nothing
    if stats === nothing || layer === nothing
        return nothing
    end
    layer.bounds_time_seconds = bounds_time_seconds
    layer.formulation_time_seconds = formulation_time_seconds
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
        :BoundOptimalCount => count_status(stats, "OPTIMAL"),
        :BoundTimeLimitCount => count_status(stats, "TIME_LIMIT"),
        :BoundIntervalArithmeticCount => count_skip(stats, "interval_arithmetic"),
        :BoundConstantExpressionCount => count_skip(stats, "constant_expression"),
        :BoundIntervalCutoffCount => count_skip(stats, "interval_proves_cutoff"),
        :BoundLowerSkippedCount => count_skip(stats, "lower_skipped_by_nonpositive_upper"),
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
