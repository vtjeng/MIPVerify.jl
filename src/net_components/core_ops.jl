using JuMP
using Memento
using MathOptInterface

"""
$(SIGNATURES)

Checks whether a JuMPLinearType is constant (and thus has no model associated)
with it. This can only be true if it is an affine expression with no stored
variables.
"""
function is_constant(x::JuMP.AffExpr)
    return iszero(x - JuMP.constant(x))
end

function is_constant(x::JuMP.VariableRef)
    false
end

function get_tightening_algorithm(
    x::JuMPLinearType,
    nta::Union{TighteningAlgorithm,Nothing},
)::TighteningAlgorithm
    if is_constant(x)
        return interval_arithmetic
    elseif !(nta === nothing)
        return nta
    else
        # x is not constant, and thus x must have an associated model
        model = owner_model(x)
        return !haskey(model.ext, :MIPVerify) ? DEFAULT_TIGHTENING_ALGORITHM :
               model.ext[:MIPVerify].tightening_algorithm
    end
end

@enum BoundType lower_bound_type = -1 upper_bound_type = 1
#! format: off
bound_f = Dict(
    lower_bound_type => lower_bound,
    upper_bound_type => upper_bound
)
bound_obj = Dict(
    lower_bound_type => MathOptInterface.MIN_SENSE,
    upper_bound_type => MathOptInterface.MAX_SENSE
)
bound_operator = Dict(
    lower_bound_type => >=,
    upper_bound_type => <=
)
bound_name = Dict(
    lower_bound_type => "lower",
    upper_bound_type => "upper"
)
#! format: on

include("lp_certification.jl")

"""
$(SIGNATURES)

Context manager for running `f` on `model`. If `should_relax_integrality` is true, the 
integrality constraints are relaxed before `f` is run and re-imposed after.
"""
function relax_integrality_context(f, model::Model, should_relax_integrality::Bool)
    if !should_relax_integrality
        return f(model)
    end

    undo_relax = relax_integrality(model)
    try
        return f(model)
    finally
        undo_relax()
    end
end

function objective_bound_or_nothing(model::JuMP.Model)
    return solver_attribute_or_nothing(
        () -> JuMP.objective_bound(model),
        "the solver objective bound",
    )
end

"""
$(SIGNATURES)

Optimizes the value of `objective` based on `bound_type`, with `b_0`, computed via interval
arithmetic, as a backup.

- If an optimal LP solution is reached and row duals are available, validate them and return the
  resulting certified Lagrangian bound. The certificate is re-verified here with outward
  rounding, so its validity does not depend on the solver's duals being exactly feasible.
  Otherwise, return `b_0`.
- If an optimal MIP solution is reached, return the solver's objective bound (its dual bound).
  In exact arithmetic, weak duality places the objective bound on the valid side of the true
  optimum, overshooting it by at most the achieved gap (objective bound minus incumbent
  objective value), and OPTIMAL termination keeps that gap within the solver's gap tolerance.
  The incumbent objective value can sit on the invalid side by up to the same gap, which is why
  it is not used here. Two claims are trusted rather than verified: (1) the reported bound is
  assembled from floating-point node relaxation bounds without directed rounding, so its
  validity is subject to the solver's numerics; (2) the incumbent is feasible only up to the
  solver's feasibility and integrality tolerances, so the true conservatism can exceed the
  reported gap by a tolerance-scale amount (this weakens the tightness claim, but a
  super-optimal incumbent cannot invalidate the bound). Solvers expose no independently
  checkable MIP certificate through JuMP, so neither claim can be re-verified the way the LP
  certificate is.
- If we reach the user-defined time limit, return `b_0`.
- For all other solve statuses, we warn the user and report `b_0`.

Whenever an optimal solution is reported, its objective value is cross-checked against `b_0`.
The solution is a feasible point of the model, so its value can never lie outside a bound that
interval arithmetic computed for the same model; if it does (beyond solver feasibility
tolerances), the solver and our view of the model disagree — solver numerics have failed or the
model plumbing is desynced — and no bound computed by this run can be trusted, so we raise an
error rather than continue.
"""
function tight_bound_helper(
    m::Model,
    bound_type::BoundType,
    objective::JuMPLinearType,
    b_0::Number,
    tightening_algorithm::TighteningAlgorithm,
    stats::Union{Nothing,VerificationStats},
)
    @objective(m, bound_obj[bound_type], objective)
    solve_start = time_ns()
    optimize!(m)
    solve_wall_time = elapsed_seconds(solve_start)
    status = JuMP.termination_status(m)
    record_bound_solve!(
        stats,
        tightening_algorithm,
        bound_name[bound_type],
        m,
        status,
        solve_wall_time,
    )
    if status == MathOptInterface.OPTIMAL
        incumbent =
            solver_attribute_or_nothing(() -> JuMP.objective_value(m), "the solver objective value")
        if incumbent !== nothing
            violation = bound_type == lower_bound_type ? b_0 - incumbent : incumbent - b_0
            if violation > 1e-8 * (1 + abs(b_0))
                Memento.error(
                    MIPVerify.LOGGER,
                    "The solver reported a feasible point with objective value $(incumbent), " *
                    "outside the interval-arithmetic bound $(b_0). The solver and the model " *
                    "disagree, so no bound computed by this run can be trusted.",
                )
            end
        end
        if tightening_algorithm == lp
            if !JuMP.has_duals(m)
                Memento.debug(
                    MIPVerify.LOGGER,
                    "Using interval-arithmetic bound: LP solver reported no dual solution.",
                )
                return b_0
            end
            return certified_lp_bound(m, bound_type, objective, b_0)
        end
        solver_bound = objective_bound_or_nothing(m)
        if solver_bound === nothing
            Memento.debug(
                MIPVerify.LOGGER,
                "Using interval-arithmetic bound: solver reported no finite objective bound.",
            )
            return b_0
        end
        return bound_type == lower_bound_type ? max(b_0, solver_bound) : min(b_0, solver_bound)
    elseif status == MathOptInterface.TIME_LIMIT
        return b_0
    else
        Memento.warn(
            MIPVerify.LOGGER,
            "Unexpected solve status $(status); using interval_arithmetic to obtain bound.",
        )
        return b_0
    end
end

"""
Calculates a tight bound of type `bound_type` on the variable `x` using the specified
tightening algorithm `nta`.

If an upper bound is proven to be below cutoff, or a lower bound is proven to above cutoff,
the algorithm returns early with whatever value was found.
"""
function tight_bound(
    x::JuMPLinearType,
    nta::Union{TighteningAlgorithm,Nothing},
    bound_type::BoundType,
    cutoff::Real,
    stats::Union{Nothing,VerificationStats} = nothing,
    resolve_stats::Bool = true,
)
    tightening_algorithm = get_tightening_algorithm(x, nta)
    b_0 = bound_f[bound_type](x)
    # Constant expressions have no owner model to resolve statistics from, so their skips
    # go unrecorded unless the caller passes `stats` explicitly (as `relu` does).
    if resolve_stats && stats === nothing && !is_constant(x)
        stats = get_verification_stats(owner_model(x))
    end
    skip_reason =
        is_constant(x) ? SKIP_CONSTANT_EXPRESSION :
        tightening_algorithm == interval_arithmetic ? SKIP_INTERVAL_ARITHMETIC :
        bound_operator[bound_type](b_0, cutoff) ? SKIP_INTERVAL_PROVES_CUTOFF : nothing
    if skip_reason !== nothing
        record_bound_skip!(stats, tightening_algorithm, bound_name[bound_type], skip_reason)
        return b_0
    end
    record_bound_request!(stats, tightening_algorithm, bound_name[bound_type])
    should_relax_integrality = (tightening_algorithm == lp)
    # x is not constant, and thus x must have an associated model
    return relax_integrality_context(owner_model(x), should_relax_integrality) do m
        tight_bound_helper(m, bound_type, x, b_0, tightening_algorithm, stats)
    end
end

function tight_upperbound(
    x::JuMPLinearType,
    stats::Union{Nothing,VerificationStats} = nothing,
    resolve_stats::Bool = true;
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
    cutoff::Real = -Inf,
)
    tight_bound(x, nta, upper_bound_type, cutoff, stats, resolve_stats)
end

function tight_lowerbound(
    x::JuMPLinearType,
    stats::Union{Nothing,VerificationStats} = nothing,
    resolve_stats::Bool = true;
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
    cutoff::Real = Inf,
)
    tight_bound(x, nta, lower_bound_type, cutoff, stats, resolve_stats)
end

function log_gap(m::JuMP.Model)
    gap = abs(1 - JuMP.objective_bound(m) / JuMP.objective_value(m))
    Memento.info(
        MIPVerify.LOGGER,
        "Hit user limit during solve to determine bounds. Multiplicative gap was $gap.",
    )
end

function relu(x::T)::T where {T<:Real}
    return max(zero(T), x)
end

function relu(x::AbstractArray{T}) where {T<:Real}
    return relu.(x)
end

function consistent_relu_bounds(x::JuMPLinearType, l::Real, u::Real)::Tuple{Real,Real}
    if u < l
        # TODO (vtjeng): This check is in place in case of numerical error in the calculation of bounds.
        # See sample number 4872 (1-indexed) when verified on the lp0.4 network.
        Memento.warn(
            MIPVerify.LOGGER,
            "Inconsistent upper and lower bounds: u-l = $(u - l) is negative. Attempting to use interval arithmetic bounds instead ...",
        )
        u = upper_bound(x)
        l = lower_bound(x)
    end
    return (l, u)
end

function relu(x::T, l::Real, u::Real)::JuMP.AffExpr where {T<:JuMPLinearType}
    (l, u) = consistent_relu_bounds(x, l, u)

    if u <= 0
        # rectified value is always 0
        return zero(T)
    elseif u == l
        return one(T) * l
    elseif u < l
        error(
            MIPVerify.LOGGER,
            "Inconsistent upper and lower bounds even after using only interval arithmetic: u-l = $(u - l) is negative",
        )
    elseif l >= 0
        # rectified value is always x
        return x
    else
        # since we know that u!=l, x is not constant, and thus x must have an associated model
        model = owner_model(x)
        x_rect = @variable(model)
        a = @variable(model, binary = true)

        # refined big-M formulation that takes advantage of the knowledge
        # that lower and upper bounds  are different.
        @constraint(model, x_rect <= x + (-l) * (1 - a))
        @constraint(model, x_rect >= x)
        @constraint(model, x_rect <= u * a)
        @constraint(model, x_rect >= 0)

        # Manually set the bounds for x_rect so they can be used by downstream operations.
        set_lower_bound(x_rect, 0)
        set_upper_bound(x_rect, u)
        return x_rect
    end
end

@enum ReLUType split = 0 zero_output = -1 linear_in_input = 1 constant_output = 2

function get_relu_type(l::Real, u::Real)::ReLUType
    if u <= 0
        return zero_output
    elseif u == l
        return constant_output
    elseif l >= 0
        return linear_in_input
    else
        return split
    end
end

struct ReLUInfo
    lowerbounds::Array{Real}
    upperbounds::Array{Real}
end

function Base.show(io::IO, s::ReLUInfo)
    relutypes = get_relu_type.(s.lowerbounds, s.upperbounds)
    print(io, "  Behavior of ReLUs - ")
    for t in instances(ReLUType)
        n = count(x -> x == t, relutypes)
        print(io, "$t: $n")
        if t != last(instances(ReLUType))
            print(io, ", ")
        end
    end
end

"""
Calculates the lower_bound only if `u` is positive; otherwise, returns `u` (since we expect)
the ReLU to be fixed to zero anyway.
"""
function lazy_tight_lowerbound(
    x::JuMPLinearType,
    u::Real,
    stats::Union{Nothing,VerificationStats} = nothing,
    resolve_stats::Bool = true;
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
    cutoff = 0,
)::Real
    if resolve_stats && stats === nothing && !is_constant(x)
        stats = get_verification_stats(owner_model(x))
    end
    if u <= cutoff
        stats === nothing || record_bound_skip!(
            stats,
            get_tightening_algorithm(x, nta),
            bound_name[lower_bound_type],
            SKIP_LOWER_SKIPPED_BY_NONPOSITIVE_UPPER,
        )
        return u
    end
    return tight_lowerbound(x, stats, false; nta = nta, cutoff = cutoff)
end

function relu(x::JuMPLinearType)::JuMP.AffExpr
    u = tight_upperbound(x, cutoff = 0)
    l = lazy_tight_lowerbound(x, u, cutoff = 0)
    relu(x, l, u)
end

"""
$(SIGNATURES)
Expresses a rectified-linearity constraint: output is constrained to be equal to
`max(x, 0)`.
"""
function relu(
    x::AbstractArray{T};
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
    stats::Union{Nothing,VerificationStats} = get_verification_stats(x),
)::Array{JuMP.AffExpr} where {T<:JuMPLinearType}
    show_progress_bar::Bool =
        MIPVerify.LOGGER.levels[MIPVerify.LOGGER.level] > MIPVerify.LOGGER.levels["debug"]

    layer_stats = if stats === nothing
        nothing
    else
        first_nonconstant_index = findfirst(x_i -> !is_constant(x_i), x)
        # A layer with no nonconstant inputs needs no bound solves, so interval arithmetic
        # is the algorithm actually applied regardless of the configured default.
        tightening_algorithm =
            first_nonconstant_index === nothing ? interval_arithmetic :
            get_tightening_algorithm(x[first_nonconstant_index], nta)
        begin_relu_layer!(stats, size(x), tightening_algorithm)
    end

    bounds_start = time_ns()
    if !show_progress_bar
        u = tight_upperbound.(x, Ref(stats), false; nta = nta, cutoff = 0)
        l = lazy_tight_lowerbound.(x, u, Ref(stats), false; nta = nta, cutoff = 0)
    else
        p1 = Progress(length(x), desc = "  Calculating upper bounds: ", enabled = isinteractive())
        u = map(x_i -> begin
            next!(p1)
            tight_upperbound(x_i, stats, false; nta = nta, cutoff = 0)
        end, x)
        p2 = Progress(length(x), desc = "  Calculating lower bounds: ", enabled = isinteractive())
        l = map(
            v -> begin
                next!(p2)
                lazy_tight_lowerbound(v..., stats, false; nta = nta, cutoff = 0)
            end,
            zip(x, u),
        )
    end
    bounds_time = elapsed_seconds(bounds_start)

    if stats !== nothing
        consistent_bounds = consistent_relu_bounds.(x, l, u)
        l = first.(consistent_bounds)
        u = last.(consistent_bounds)
    end

    if show_progress_bar
        reluinfo = ReLUInfo(l, u)
        Memento.info(MIPVerify.LOGGER, "$reluinfo")
    end

    constraint_start = time_ns()
    if !show_progress_bar
        x_r = relu.(x, l, u)
    else
        p3 = Progress(length(x), desc = "  Imposing relu constraint: ", enabled = isinteractive())
        x_r = map(v -> (next!(p3); relu(v...)), zip(x, l, u))
    end
    if stats !== nothing
        relutypes = get_relu_type.(l, u)
        finish_relu_layer!(
            stats,
            layer_stats,
            bounds_time,
            elapsed_seconds(constraint_start),
            count(==(zero_output), relutypes),
            count(==(linear_in_input), relutypes),
            count(==(constant_output), relutypes),
            count(==(split), relutypes),
        )
    end
    return x_r
end

function masked_relu(x::T, m::Real)::T where {T<:Real}
    if m < 0
        zero(T)
    elseif m > 0
        x
    else
        relu(x)
    end
end

function masked_relu(x::AbstractArray{<:Real}, m::AbstractArray{<:Real})
    masked_relu.(x, m)
end

function masked_relu(x::T, m::Real)::JuMP.AffExpr where {T<:JuMPLinearType}
    if m < 0
        zero(T)
    elseif m > 0
        x
    else
        relu(x)
    end
end

"""
$(SIGNATURES)
Expresses a masked rectified-linearity constraint, with three possibilities depending on
the value of the mask. Output is constrained to be:
```
1) max(x, 0) if m=0,
2) 0 if m<0
3) x if m>0
```
"""
function masked_relu(
    x::AbstractArray{<:JuMPLinearType},
    m::AbstractArray{<:Real};
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
)::Array{JuMP.AffExpr}
    @assert(size(x) == size(m))
    s = size(m)
    stats = get_verification_stats(x)
    # We add the constraints corresponding to the active ReLUs to the model
    zero_idx = Iterators.filter(i -> m[i] == 0, CartesianIndices(s)) |> collect
    d = Dict(zip(zero_idx, relu(x[zero_idx], nta = nta, stats = stats)))

    # We determine the output of the masked relu, which is either:
    #  1) the output of the relu that we have previously determined when adding the
    #     constraints to the model.
    #  2, 3) the result of applying the (elementwise) masked_relu function.
    output = map(i -> m[i] == 0 ? d[i] : masked_relu(x[i], m[i]), CartesianIndices(s))
    if stats !== nothing
        # `relu` recorded a layer covering only the unmasked entries; widen it to the full
        # layer shape and add the entries whose phase the mask fixes.
        layer_stats = stats.relu_layers[end]
        layer_stats.input_shape = size(x)
        layer_stats.num_zero_output += count(<(0), m)
        layer_stats.num_linear_in_input += count(>(0), m)
    end
    return output
end

function maximum(xs::AbstractArray{T})::T where {T<:Real}
    return Base.maximum(xs)
end

function maximum_of_constants(xs::AbstractArray{T}) where {T<:JuMPLinearType}
    @assert all(is_constant.(xs))
    max_val = map(x -> x.constant, xs) |> maximum
    return one(JuMP.VariableRef) * max_val
end

function log_maximum_candidate_count(model::Model, num_candidates::Integer, num_inputs::Integer)
    counter_key = :MIPVerifyMaximumCallCount
    call_count = get(model.ext, counter_key, 0) + 1
    model.ext[counter_key] = call_count

    log_message = "Maximum call #$(call_count): $(num_candidates) of $(num_inputs) inputs can still attain the maximum."
    if call_count == 1
        Memento.info(MIPVerify.LOGGER, log_message)
    else
        Memento.debug(MIPVerify.LOGGER, log_message)
    end
end

"""
$(SIGNATURES)
Expresses a maximization constraint: output is constrained to be equal to `max(xs)`.
"""
function maximum(xs::AbstractArray{T})::JuMP.AffExpr where {T<:JuMPLinearType}
    if length(xs) == 1
        return xs[1]
    end

    if all(is_constant.(xs))
        return maximum_of_constants(xs)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)

    p1 = Progress(length(xs), desc = "  Calculating upper bounds: ", enabled = isinteractive())
    us = map(x_i -> (next!(p1); tight_upperbound(x_i)), xs)
    p2 = Progress(length(xs), desc = "  Calculating lower bounds: ", enabled = isinteractive())
    ls = map(x_i -> (next!(p2); tight_lowerbound(x_i)), xs)

    l = Base.maximum(ls)
    u = Base.maximum(us)

    if l == u
        return one(T) * l
        Memento.info(MIPVerify.LOGGER, "Output of maximum is constant.")
    end
    # at least one index will satisfy this property because of check above.
    active_indexes = findall(us .> l)
    log_maximum_candidate_count(model, length(active_indexes), length(xs))

    return maximum(xs[active_indexes], ls[active_indexes], us[active_indexes])
end

function maximum(
    xs::AbstractArray{T,1},
    ls::AbstractArray{<:Real,1},
    us::AbstractArray{<:Real,1},
)::JuMP.AffExpr where {T<:JuMPLinearType}

    @assert length(xs) > 0
    @assert length(xs) == length(ls)
    @assert length(xs) == length(us)

    if all(is_constant.(xs))
        return maximum_of_constants(xs)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)
    if length(xs) == 1
        return first(xs)
    else
        l = Base.maximum(ls)
        u = Base.maximum(us)
        x_max = @variable(model, lower_bound = l, upper_bound = u)
        a = @variable(model, [1:length(xs)], binary = true)
        @constraint(model, sum(a) == 1)
        for (i, x) in enumerate(xs)
            umaxi = Base.maximum(us[1:end.!=i])
            @constraint(model, x_max <= x + (1 - a[i]) * (umaxi - ls[i]))
            @constraint(model, x_max >= x)
        end
        return x_max
    end
end

"""
$(SIGNATURES)
Expresses a one-sided maximization constraint: output is constrained to be at least
`max(xs)`.

Only use when you are minimizing over the output in the objective.

NB: If all of xs are constant, we simply return the largest of them.
"""
function maximum_ge(xs::AbstractArray{T})::JuMPLinearType where {T<:JuMPLinearType}
    @assert length(xs) > 0
    if all(is_constant.(xs))
        return maximum_of_constants(xs)
    end
    if length(xs) == 1
        return first(xs)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)
    x_max = @variable(model)
    @constraint(model, x_max .>= xs)
    # In the standard flow x_max is only created after bound tightening has completed, but
    # declare the lower bound implied by the constraints above anyway: if a variable with no
    # finite declared bounds is present in a model during a later `certified_lp_bound` call,
    # any nonzero stationarity residual on it forces the certificate to be abandoned.
    implied_lower = maximum(lower_bound.(xs))
    isfinite(implied_lower) && set_lower_bound(x_max, implied_lower)
    return x_max
end

"""
$(SIGNATURES)
Expresses a one-sided absolute-value constraint: output is constrained to be at least as
large as `|x|`.

Only use when you are minimizing over the output in the objective.
"""
function abs_ge(x::JuMPLinearType)::JuMP.AffExpr
    if is_constant(x)
        return one(JuMP.VariableRef) * abs(x.constant)
    end
    model = owner_model(x)
    u = upper_bound(x)
    l = lower_bound(x)
    if u <= 0
        return -x
    elseif l >= 0
        return x
    else
        x_abs = @variable(model)
        @constraint(model, x_abs >= x)
        @constraint(model, x_abs >= -x)
        set_lower_bound(x_abs, 0)
        set_upper_bound(x_abs, max(-l, u))
        return x_abs
    end
end

function get_target_indexes(
    target_index::Integer,
    array_length::Integer;
    invert_target_selection::Bool = false,
)

    get_target_indexes(
        [target_index],
        array_length,
        invert_target_selection = invert_target_selection,
    )

end

function get_target_indexes(
    target_indexes::Array{<:Integer,1},
    array_length::Integer;
    invert_target_selection::Bool = false,
)::AbstractArray{<:Integer,1}

    @assert length(target_indexes) >= 1
    @assert all(target_indexes .>= 1) && all(target_indexes .<= array_length)

    invert_target_selection ? filter((x) -> x ∉ target_indexes, 1:array_length) : target_indexes
end

function get_vars_for_max_index(
    xs::Array{<:JuMPLinearType,1},
    target_indexes::Array{<:Integer,1},
)::Tuple{JuMPLinearType,Array{<:JuMPLinearType,1}}

    @assert length(xs) >= 1

    target_vars = xs[Bool[i ∈ target_indexes for i in 1:length(xs)]]
    nontarget_vars = xs[Bool[i ∉ target_indexes for i in 1:length(xs)]]

    maximum_target_var = length(target_vars) == 1 ? target_vars[1] : MIPVerify.maximum(target_vars)

    return (maximum_target_var, nontarget_vars)
end

"""
$(SIGNATURES)

Imposes constraints ensuring that one of the elements at the target_indexes is (tied for) the
largest element of the array x. More specifically, we require `x[j] - x[i] ≥ margin` for
some `j ∈ target_indexes` and for all `i ∉ target_indexes`.
"""
function set_max_indexes(
    model::Model,
    xs::Array{<:JuMPLinearType,1},
    target_indexes::Array{<:Integer,1};
    margin::Real = 0,
)::Nothing

    (maximum_target_var, nontarget_vars) = get_vars_for_max_index(xs, target_indexes)

    # JuMP does not support strict inequalities; see 
    # https://github.com/jump-dev/JuMP.jl/blob/24c0409c5fa5cae6a4ae64b1c82ab5f83d55fbc6/src/macros/%40variable.jl#L516-L523
    # for more context.
    @constraint(model, nontarget_vars .<= maximum_target_var - margin)
    return nothing
end
