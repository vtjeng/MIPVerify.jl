module PGDWarmStart

using HiGHS
using JuMP
using MIPVerify
using NNlib
using Random
using SHA
using Zygote

const MOI = JuMP.MOI

export PGDResult,
    WarmStartVariant,
    WorstMarginProblem,
    FullStart,
    project_linf,
    inward_project_linf,
    uniform_box_start,
    optimized_logits,
    projected_gradient_attack,
    worst_margin,
    build_worst_margin_problem,
    copy_problem,
    apply_sparse_start!,
    complete_full_start,
    apply_full_start!,
    solve_worst_margin!,
    verify_candidate,
    model_signature,
    process_rss_mb,
    system_available_mb,
    require_available_memory,
    WARM_START_VARIANTS,
    DIAGNOSTIC_WARM_START_VARIANTS,
    ordered_variants,
    apply_variant_start!

struct PGDResult
    candidate::Array{Float64,4}
    margin::Float64
    competing_index::Int
    restart::Int
    step::Int
    elapsed_seconds::Float64
    restart_best_margins::Vector{Float64}
end

struct ModelSignature
    num_variables::Int
    num_binary_variables::Int
    num_structural_constraints::Int
    variable_bounds_sha256::String
end

Base.:(==)(left::ModelSignature, right::ModelSignature) =
    left.num_variables == right.num_variables &&
    left.num_binary_variables == right.num_binary_variables &&
    left.num_structural_constraints == right.num_structural_constraints &&
    left.variable_bounds_sha256 == right.variable_bounds_sha256

"""A benchmark treatment whose name and start coverage are part of the output schema."""
struct WarmStartVariant
    name::Symbol
    source::Symbol
    coverage::Symbol
end

const WARM_START_VARIANTS = (
    # No start values: this is the primary control.
    WarmStartVariant(:cold, :none, :none),
    # A deterministic unselected point controls for complete-start mechanics.
    WarmStartVariant(:random_full, :random, :all_variables),
    # Every MIP variable starts from a feasible completion of the PGD input.
    WarmStartVariant(:pgd_full, :pgd, :all_variables),
)

const DIAGNOSTIC_WARM_START_VARIANTS = (
    # Input and perturbation starts from the unperturbed image diagnose generic start effects.
    WarmStartVariant(:original_sparse, :original, :input_and_perturbation),
    # Input and perturbation starts from PGD diagnose partial-start completion.
    WarmStartVariant(:pgd_sparse, :pgd, :input_and_perturbation),
    # The unperturbed input is retained as a complete-start diagnostic.
    WarmStartVariant(:original_full, :original, :all_variables),
)

function variant_by_name(name::Symbol)::WarmStartVariant
    variants = (WARM_START_VARIANTS..., DIAGNOSTIC_WARM_START_VARIANTS...)
    index = findfirst(variant -> variant.name == name, variants)
    isnothing(index) && throw(ArgumentError("unknown warm-start variant: $name"))
    return variants[index]
end

"""Rotate treatment order between blocks without changing the set of paired treatments."""
function ordered_variants(block_id::Integer)
    block_id > 0 || throw(ArgumentError("block_id must be positive"))
    offset = mod(block_id - 1, length(WARM_START_VARIANTS))
    return ntuple(
        index -> WARM_START_VARIANTS[mod1(index + offset, length(WARM_START_VARIANTS))],
        length(WARM_START_VARIANTS),
    )
end

struct WorstMarginProblem
    model::JuMP.Model
    perturbed_input::Any
    perturbation::Any
    output::Any
    margin_variable::JuMP.VariableRef
    true_index::Int
    formulation_time_seconds::Float64
    signature::ModelSignature
    formulation_stats::Dict{Symbol,Any}
end

struct FullStart
    values_by_index::Dict{Int64,Float64}
    completion_time_seconds::Float64
    termination_status::MOI.TerminationStatusCode
end

function process_rss_mb()::Float64
    return try
        line = only(filter(line -> startswith(line, "VmRSS:"), readlines("/proc/self/status")))
        parse(Float64, split(line)[2]) / 1024
    catch
        NaN
    end
end


function system_available_mb()::Float64
    return try
        line = only(filter(line -> startswith(line, "MemAvailable:"), readlines("/proc/meminfo")))
        parse(Float64, split(line)[2]) / 1024
    catch
        NaN
    end
end


function require_available_memory(
    minimum_mb::Real;
    available_mb::Real = system_available_mb(),
)::Float64
    minimum_mb > 0 || throw(ArgumentError("minimum_mb must be positive"))
    if isfinite(available_mb) && available_mb < minimum_mb
        error(
            "only $(round(available_mb; digits = 1)) MiB is available; " *
            "the serial warm-start benchmark requires at least $minimum_mb MiB",
        )
    end
    return Float64(available_mb)
end

function project_linf(candidate, input, epsilon::Real)
    size(candidate) == size(input) || throw(DimensionMismatch("candidate and input must match"))
    epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    lower = max.(0.0, Float64.(input) .- epsilon)
    upper = min.(1.0, Float64.(input) .+ epsilon)
    return clamp.(Float64.(candidate), lower, upper)
end

function inward_project_linf(
    candidate,
    input,
    epsilon::Real;
    absolute_margin::Real = 1e-9,
    relative_margin::Real = 1e-8,
)
    epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    absolute_margin >= 0 || throw(ArgumentError("absolute_margin must be nonnegative"))
    relative_margin >= 0 || throw(ArgumentError("relative_margin must be nonnegative"))
    margin = min(epsilon, max(absolute_margin, epsilon * relative_margin))
    return project_linf(candidate, input, epsilon - margin)
end

function uniform_box_start(rng::AbstractRNG, lower, upper)
    size(lower) == size(upper) || throw(DimensionMismatch("box bounds must match"))
    all(lower .<= upper) || throw(ArgumentError("box lower bound exceeds upper bound"))
    return Float64.(lower) .+ rand(rng, size(lower)...) .* (Float64.(upper) .- Float64.(lower))
end

function competing_index(logits::AbstractVector{<:Real}, true_index::Int)::Int
    checkbounds(logits, true_index)
    length(logits) > 1 || throw(ArgumentError("at least two logits are required"))
    best = true_index == 1 ? 2 : 1
    for index in eachindex(logits)
        if index != true_index && logits[index] > logits[best]
            best = index
        end
    end
    return best
end

function worst_margin(logits::AbstractVector{<:Real}, true_index::Int)::Float64
    other = competing_index(logits, true_index)
    return Float64(logits[other] - logits[true_index])
end

function supports_optimized_pgd(nn::MIPVerify.Sequential)::Bool
    return length(nn.layers) == 8 &&
           nn.layers[1] isa MIPVerify.Conv2d &&
           nn.layers[2] isa MIPVerify.ReLU &&
           nn.layers[3] isa MIPVerify.Conv2d &&
           nn.layers[4] isa MIPVerify.ReLU &&
           nn.layers[5] isa MIPVerify.Flatten &&
           collect(nn.layers[5].perm) == [1, 3, 2, 4] &&
           nn.layers[6] isa MIPVerify.Linear &&
           nn.layers[7] isa MIPVerify.ReLU &&
           nn.layers[8] isa MIPVerify.Linear
end

function same_padding(layer::MIPVerify.Conv2d, input_width::Int, input_height::Int)
    filter_height, filter_width = size(layer.filter)[1:2]
    output_width = cld(input_width, layer.stride)
    output_height = cld(input_height, layer.stride)
    total_width = max((output_width - 1) * layer.stride + filter_width - input_width, 0)
    total_height = max((output_height - 1) * layer.stride + filter_height - input_height, 0)
    left = fld(total_width, 2)
    top = fld(total_height, 2)
    return (left, total_width - left, top, total_height - top)
end

function optimized_conv(input_whcn, layer::MIPVerify.Conv2d)
    filter = Float64.(permutedims(layer.filter, (2, 1, 3, 4)))
    padding = if layer.padding isa MIPVerify.SamePadding
        same_padding(layer, size(input_whcn, 1), size(input_whcn, 2))
    elseif layer.padding isa MIPVerify.ValidPadding
        0
    else
        top, bottom, left, right = MIPVerify.compute_padding_values(layer.padding)
        (left, right, top, bottom)
    end
    output = NNlib.conv(
        input_whcn,
        filter;
        stride = (layer.stride, layer.stride),
        pad = padding,
        # MIPVerify's stored kernels match this NNlib convention; parity tests cover the even,
        # strided convolution used by WK17a.
        flipped = true,
    )
    return output .+ reshape(Float64.(layer.bias), 1, 1, :, 1)
end

"""Evaluate the WK17a-style network on `(width, height, channel, batch)` inputs."""
function optimized_logits(nn::MIPVerify.Sequential, input_whcn)
    supports_optimized_pgd(nn) ||
        throw(ArgumentError("optimized PGD requires the WK17a-style 8-layer architecture"))
    value = max.(optimized_conv(input_whcn, nn.layers[1]), 0.0)
    value = max.(optimized_conv(value, nn.layers[3]), 0.0)
    features = reshape(value, :, size(value, 4))
    hidden = transpose(Float64.(nn.layers[6].matrix)) * features .+ Float64.(nn.layers[6].bias)
    hidden = max.(hidden, 0.0)
    return transpose(Float64.(nn.layers[8].matrix)) * hidden .+ Float64.(nn.layers[8].bias)
end

function batched_margins(logits::AbstractMatrix, true_index::Int)
    others = filter(!=(true_index), collect(axes(logits, 1)))
    return vec(maximum(logits[others, :]; dims = 1)) .- vec(logits[true_index, :])
end

"""Run batched, deterministic projected gradient descent and retain every trajectory's best."""
function projected_gradient_attack(
    nn::MIPVerify.Sequential,
    input::Array{<:Real,4},
    true_index::Int;
    epsilon::Real = 0.1,
    step_size::Real = 0.01,
    steps::Int = 100,
    restarts::Int = 20,
    seed::Integer = 0,
)
    supports_optimized_pgd(nn) || throw(ArgumentError("unsupported network architecture"))
    epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    step_size > 0 || throw(ArgumentError("step_size must be positive"))
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    restarts > 0 || throw(ArgumentError("restarts must be positive"))

    started_at = time_ns()
    input_whcn = permutedims(Float64.(input), (3, 2, 4, 1))
    size(input_whcn, 4) == 1 || throw(ArgumentError("PGD expects one sample"))
    lower = max.(0.0, input_whcn .- epsilon)
    upper = min.(1.0, input_whcn .+ epsilon)
    rng = MersenneTwister(seed)
    candidates = uniform_box_start(
        rng,
        repeat(lower; outer = (1, 1, 1, restarts)),
        repeat(upper; outer = (1, 1, 1, restarts)),
    )
    candidates[:, :, :, 1:1] .= input_whcn
    best_candidates = copy(candidates)
    best_margins = fill(-Inf, restarts)
    best_steps = zeros(Int, restarts)
    best_competitors = zeros(Int, restarts)

    for step in 0:steps
        logits = optimized_logits(nn, candidates)
        margins = batched_margins(logits, true_index)
        for restart in 1:restarts
            if margins[restart] > best_margins[restart]
                best_margins[restart] = margins[restart]
                best_steps[restart] = step
                best_candidates[:, :, :, restart] .= candidates[:, :, :, restart]
                best_competitors[restart] = competing_index(view(logits, :, restart), true_index)
            end
        end
        maximum(best_margins) >= 0 && break
        step == steps && break
        gradient = only(Zygote.gradient(candidates) do values
            sum(batched_margins(optimized_logits(nn, values), true_index))
        end)
        candidates = clamp.(candidates .+ step_size .* sign.(gradient), lower, upper)
    end

    best_restart = argmax(best_margins)
    candidate = permutedims(best_candidates[:, :, :, best_restart:best_restart], (4, 2, 1, 3))
    return PGDResult(
        Array(candidate),
        best_margins[best_restart],
        best_competitors[best_restart],
        best_restart,
        best_steps[best_restart],
        (time_ns() - started_at) / 1e9,
        best_margins,
    )
end

function model_signature(model::JuMP.Model)::ModelSignature
    io = IOBuffer()
    for variable in JuMP.all_variables(model)
        print(io, JuMP.index(variable).value, ':')
        lower = JuMP.has_lower_bound(variable) ? JuMP.lower_bound(variable) : -Inf
        upper = JuMP.has_upper_bound(variable) ? JuMP.upper_bound(variable) : Inf
        print(io, bitstring(Float64(lower)), ':', bitstring(Float64(upper)), '\n')
    end
    return ModelSignature(
        JuMP.num_variables(model),
        JuMP.num_constraints(model, JuMP.VariableRef, MOI.ZeroOne),
        JuMP.num_constraints(model; count_variable_in_set_constraints = false),
        bytes2hex(sha256(take!(io))),
    )
end

function build_worst_margin_problem(
    nn::MIPVerify.NeuralNet,
    input::Array{<:Real},
    true_index::Int,
    optimizer,
    tightening_options::Dict;
    pp::MIPVerify.PerturbationFamily,
    tightening_algorithm::MIPVerify.TighteningAlgorithm,
)
    started_at = time_ns()
    data = MIPVerify.get_model(
        nn,
        input,
        pp,
        optimizer,
        tightening_options,
        tightening_algorithm,
        true,
    )
    output = vec(data[:Output])
    others = filter(!=(true_index), collect(eachindex(output)))
    # The maximum is itself maximized, so this must be the exact selector formulation. The
    # one-sided maximum_ge helper is only valid when its output is minimized.
    maximum_other = MIPVerify.maximum(output[others])
    margin_variable = @variable(data[:Model])
    @constraint(data[:Model], margin_variable == maximum_other - output[true_index])
    @objective(data[:Model], Max, margin_variable)
    stats = MIPVerify.summarize_verification_stats(data[:Model])
    empty!(data[:Model].ext)
    elapsed = (time_ns() - started_at) / 1e9
    return WorstMarginProblem(
        data[:Model],
        data[:PerturbedInput],
        data[:Perturbation],
        data[:Output],
        margin_variable,
        true_index,
        elapsed,
        model_signature(data[:Model]),
        stats,
    )
end

mapped_array(reference_map, values) = map(value -> reference_map[value], values)

function copy_problem(problem::WorstMarginProblem)::WorstMarginProblem
    model, reference_map = JuMP.copy_model(problem.model)
    copied = WorstMarginProblem(
        model,
        mapped_array(reference_map, problem.perturbed_input),
        mapped_array(reference_map, problem.perturbation),
        mapped_array(reference_map, problem.output),
        reference_map[problem.margin_variable],
        problem.true_index,
        problem.formulation_time_seconds,
        model_signature(model),
        problem.formulation_stats,
    )
    copied.signature == problem.signature || error("copied model signature differs from base")
    return copied
end

function apply_sparse_start!(problem, candidate, input)
    size(candidate) == size(problem.perturbed_input) || throw(DimensionMismatch("candidate"))
    for index in eachindex(candidate)
        JuMP.set_start_value(problem.perturbed_input[index], Float64(candidate[index]))
        JuMP.set_start_value(problem.perturbation[index], Float64(candidate[index] - input[index]))
    end
    return nothing
end

function set_serial_highs!(model, time_limit; output_flag = false)
    JuMP.set_optimizer(model, HiGHS.Optimizer)
    JuMP.set_optimizer_attribute(model, "threads", 1)
    JuMP.set_optimizer_attribute(model, "parallel", "off")
    JuMP.set_optimizer_attribute(model, "random_seed", 0)
    JuMP.set_optimizer_attribute(model, "presolve", "choose")
    JuMP.set_optimizer_attribute(model, "time_limit", Float64(time_limit))
    JuMP.set_optimizer_attribute(model, "output_flag", output_flag)
    return nothing
end

function complete_full_start(problem, candidate; time_limit = 30.0)
    completion = copy_problem(problem)
    for index in eachindex(candidate)
        JuMP.fix(completion.perturbed_input[index], Float64(candidate[index]); force = true)
    end
    JuMP.set_objective_sense(completion.model, MOI.FEASIBILITY_SENSE)
    set_serial_highs!(completion.model, time_limit)
    started_at = time_ns()
    JuMP.optimize!(completion.model)
    elapsed = (time_ns() - started_at) / 1e9
    status = JuMP.termination_status(completion.model)
    JuMP.primal_status(completion.model) == MOI.FEASIBLE_POINT ||
        error("full-start completion failed: $status")
    values = Dict(
        Int64(JuMP.index(variable).value) => Float64(JuMP.value(variable)) for
        variable in JuMP.all_variables(completion.model)
    )
    return FullStart(values, elapsed, status)
end

function apply_full_start!(problem, full_start::FullStart)
    variables = JuMP.all_variables(problem.model)
    length(variables) == length(full_start.values_by_index) || error("incomplete full start")
    for variable in variables
        JuMP.set_start_value(
            variable,
            full_start.values_by_index[Int64(JuMP.index(variable).value)],
        )
    end
    return nothing
end

function apply_variant_start!(
    problem,
    variant::WarmStartVariant,
    original,
    pgd_candidate;
    full_start::Union{Nothing,FullStart} = nothing,
)
    if variant.name == :cold
        return nothing
    elseif variant.name == :original_sparse
        apply_sparse_start!(problem, original, original)
    elseif variant.name == :pgd_sparse
        apply_sparse_start!(problem, pgd_candidate, original)
    elseif variant.name in (:original_full, :random_full, :pgd_full)
        isnothing(full_start) &&
            throw(ArgumentError("$(variant.name) requires a completed full start"))
        apply_full_start!(problem, full_start)
    else
        throw(ArgumentError("unsupported warm-start variant: $(variant.name)"))
    end
    return nothing
end

function apply_variant_start!(problem, name::Symbol, original, pgd_candidate; kwargs...)
    return apply_variant_start!(problem, variant_by_name(name), original, pgd_candidate; kwargs...)
end

function verify_candidate(nn, input, candidate, pp, true_index)
    output = vec(candidate |> nn)
    targets = filter(!=(true_index), collect(eachindex(output)))
    margin, target_verified = MIPVerify.witness_satisfies_target(output, targets, 0.0)
    perturbation_verified, _ =
        MIPVerify.verify_perturbation_witness(pp, input, candidate, Dict{Symbol,Any}())
    return (
        margin = Float64(margin),
        target_verified = target_verified,
        perturbation_verified = perturbation_verified,
        verified_attack = target_verified && perturbation_verified,
        output = Float64.(output),
    )
end

mutable struct SolveTrace
    rows::Vector{NamedTuple}
    stopped_on_negative_bound::Bool
end

function trace_callback!(trace, tolerance, callback_type, ::Ptr{Cchar}, data)::Cint
    if callback_type in (HiGHS.kHighsCallbackMipLogging, HiGHS.kHighsCallbackMipImprovingSolution)
        push!(
            trace.rows,
            (
                event = callback_type == HiGHS.kHighsCallbackMipLogging ? "progress" :
                        "improving_solution",
                solver_time_seconds = Float64(data.running_time),
                node_count = Int64(data.mip_node_count),
                simplex_iterations = Int64(data.mip_total_lp_iterations),
                primal_bound = Float64(data.mip_primal_bound),
                dual_bound = Float64(data.mip_dual_bound),
                relative_gap = Float64(data.mip_gap),
            ),
        )
    elseif callback_type == HiGHS.kHighsCallbackMipInterrupt &&
           isfinite(data.mip_dual_bound) &&
           data.mip_dual_bound < -tolerance
        trace.stopped_on_negative_bound = true
        return Cint(1)
    end
    return Cint(0)
end

safe_metric(f, model, fallback) =
    try
        f(model)
    catch
        fallback
    end

function solve_worst_margin!(
    problem,
    nn,
    input,
    pp;
    time_limit,
    robust_tolerance = 1e-8,
    log_path = nothing,
)
    trace = SolveTrace(NamedTuple[], false)
    set_serial_highs!(problem.model, time_limit; output_flag = log_path !== nothing)
    if log_path !== nothing
        mkpath(dirname(log_path))
        JuMP.set_optimizer_attribute(problem.model, "log_to_console", false)
        JuMP.set_optimizer_attribute(problem.model, "log_file", log_path)
        JuMP.set_optimizer_attribute(problem.model, "mip_min_logging_interval", 1.0)
    end
    attribute = HiGHS.CallbackFunction(
        Cint[
            HiGHS.kHighsCallbackIpmInterrupt,
            HiGHS.kHighsCallbackMipInterrupt,
            HiGHS.kHighsCallbackMipLogging,
            HiGHS.kHighsCallbackMipImprovingSolution,
        ],
    )
    callback =
        (kind, message, data) -> trace_callback!(trace, robust_tolerance, kind, message, data)
    JuMP.set_attribute(problem.model, attribute, callback)
    JuMP.set_attribute(problem.model, MOI.ObjectiveLimit(), 0.0)
    rss_before = process_rss_mb()
    started_at = time_ns()
    JuMP.optimize!(problem.model)
    wall_time = (time_ns() - started_at) / 1e9
    rss_after = process_rss_mb()
    termination = JuMP.termination_status(problem.model)
    primal = JuMP.primal_status(problem.model)
    bound = safe_metric(JuMP.objective_bound, problem.model, Inf)
    value =
        primal == MOI.FEASIBLE_POINT ? safe_metric(JuMP.objective_value, problem.model, missing) :
        missing
    candidate =
        primal == MOI.FEASIBLE_POINT ? Array(Float64.(JuMP.value.(problem.perturbed_input))) :
        nothing
    verification =
        candidate === nothing ? nothing :
        verify_candidate(nn, input, candidate, pp, problem.true_index)
    certified =
        isfinite(bound) &&
        bound < -robust_tolerance &&
        !(verification !== nothing && verification.verified_attack)
    outcome = if verification !== nothing && verification.verified_attack
        "verified_attack"
    elseif certified
        "certified_robust"
    elseif termination == MOI.TIME_LIMIT
        "time_limit_unresolved"
    else
        "unresolved"
    end
    return (
        outcome = outcome,
        termination_status = string(termination),
        primal_status = string(primal),
        objective_value = value,
        objective_bound = bound,
        candidate = candidate,
        verification = verification,
        stopped_on_negative_bound = trace.stopped_on_negative_bound,
        wall_time_seconds = wall_time,
        solver_time_seconds = safe_metric(JuMP.solve_time, problem.model, missing),
        node_count = safe_metric(JuMP.node_count, problem.model, missing),
        simplex_iterations = safe_metric(JuMP.simplex_iterations, problem.model, missing),
        relative_gap = safe_metric(JuMP.relative_gap, problem.model, missing),
        rss_before_mb = rss_before,
        rss_after_mb = rss_after,
        max_rss_mb = Sys.maxrss() / 2.0^20,
        trace = trace,
    )
end

end
