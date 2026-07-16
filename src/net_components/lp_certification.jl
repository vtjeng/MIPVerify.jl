const ZERO_INTERVAL = IntervalArithmetic.interval(0.0)

function add_interval_coefficient!(coefficients, variable::JuMP.VariableRef, delta)
    coefficients[variable] = get(coefficients, variable, ZERO_INTERVAL) + delta
    return nothing
end

function projected_dual_and_reference(set::MathOptInterface.LessThan, dual_value::Real)
    return (min(dual_value, 0.0), set.upper)
end

function projected_dual_and_reference(set::MathOptInterface.GreaterThan, dual_value::Real)
    return (max(dual_value, 0.0), set.lower)
end

function projected_dual_and_reference(set::MathOptInterface.EqualTo, dual_value::Real)
    return (dual_value, set.value)
end

function projected_dual_and_reference(set::MathOptInterface.Interval, dual_value::Real)
    reference = dual_value >= 0 ? set.lower : set.upper
    return (dual_value, reference)
end

projected_dual_and_reference(::MathOptInterface.AbstractScalarSet, ::Real) = nothing

"""
    solver_read_or_nothing(f, log_unexpected, description, consequence)

Run `f` and return its result, or `nothing` if the read fails.

Reading an optional solver attribute (a row dual, an objective bound) can tighten a bound but
is never required for soundness, so the caller always has a valid fallback. The three MOI
errors that signal an unavailable attribute are expected and return `nothing` quietly. Any
other error is logged via `log_unexpected` and also returns `nothing`, so one failed read
degrades a single bound instead of crashing the run; interrupts and resource exhaustion still
propagate.
"""
function solver_read_or_nothing(
    f,
    log_unexpected,
    description::AbstractString,
    consequence::AbstractString,
)
    try
        return f()
    catch error
        if error isa InterruptException ||
           error isa OutOfMemoryError ||
           error isa StackOverflowError
            rethrow()
        end
        if !(
            error isa MathOptInterface.ResultIndexBoundsError ||
            error isa MathOptInterface.UnsupportedAttribute ||
            error isa MathOptInterface.GetAttributeNotAllowed
        )
            log_unexpected(
                MIPVerify.LOGGER,
                "Unexpected error reading $(description); $(consequence): " *
                sprint(showerror, error),
            )
        end
        return nothing
    end
end

is_finite_real(value) = value isa Real && isfinite(value)

"""
    solver_attribute_or_nothing(f, description)

Run `f` and return its result if it is a finite `Real`, and `nothing` otherwise.
"""
function solver_attribute_or_nothing(f, description::AbstractString)
    value = solver_read_or_nothing(f, Memento.warn, description, "treating it as unavailable")
    return is_finite_real(value) ? value : nothing
end

function constraint_dual_or_nothing(constraint, dual_value)
    return solver_read_or_nothing(
        () -> dual_value(constraint),
        Memento.warn,
        "a constraint dual",
        "treating it as unavailable",
    )
end

function default_constraint_duals(model::JuMP.Model, constraints)
    JuMP.has_duals(model) || return nothing
    return MathOptInterface.get(
        JuMP.backend(model),
        MathOptInterface.ConstraintDual(),
        JuMP.index.(constraints),
    )
end

function constraint_duals_or_nothing(constraints, dual_values)
    values = solver_read_or_nothing(
        () -> dual_values(constraints),
        Memento.debug,
        "a batch of constraint duals",
        "retrying individually",
    )
    values === nothing && return nothing
    if !(values isa AbstractVector) || length(values) != length(constraints)
        Memento.debug(
            MIPVerify.LOGGER,
            "Batch constraint-dual read returned an incompatible value; retrying individually.",
        )
        return nothing
    end
    return values
end

function variable_interval_or_nothing(variable::JuMP.VariableRef)
    if JuMP.is_fixed(variable)
        value = JuMP.fix_value(variable)
        return isfinite(value) ? IntervalArithmetic.interval(value) : nothing
    end
    lower = JuMP.has_lower_bound(variable) ? JuMP.lower_bound(variable) : -Inf
    upper = JuMP.has_upper_bound(variable) ? JuMP.upper_bound(variable) : Inf
    if JuMP.is_binary(variable)
        lower = max(lower, 0.0)
        upper = min(upper, 1.0)
    end
    if isnan(lower) || isnan(upper) || lower > upper
        return nothing
    end
    return IntervalArithmetic.interval(lower, upper)
end

function is_usable_constraint_dual(row_dual)
    return is_finite_real(row_dual) && !iszero(row_dual)
end

"""
    constraint_certificate_term!(coefficients, constraint, row_dual)

Apply one row's certificate contribution, split in two: the row's variable terms are subtracted
into `coefficients` (mutated in place), and its scalar `multiplier * (reference - constant)`
term is returned. Returns `nothing`, leaving `coefficients` unchanged, when the row's set is
unsupported or the projected multiplier or reference is unusable.
"""
function constraint_certificate_term!(coefficients, constraint, row_dual::Real)
    constraint_object = JuMP.constraint_object(constraint)
    projected = projected_dual_and_reference(constraint_object.set, row_dual)
    projected === nothing && return nothing
    multiplier, reference = projected
    (iszero(multiplier) || !isfinite(reference)) && return nothing
    multiplier_interval = IntervalArithmetic.interval(multiplier)
    function_value = constraint_object.func
    for (variable, coefficient) in function_value.terms
        add_interval_coefficient!(
            coefficients,
            variable,
            -multiplier_interval * IntervalArithmetic.interval(coefficient),
        )
    end
    return multiplier_interval * (
        IntervalArithmetic.interval(reference) -
        IntervalArithmetic.interval(function_value.constant)
    )
end

function add_constraint_duals_to_certificate!(coefficients, certificate, constraints, row_duals)
    for (constraint, row_dual) in zip(constraints, row_duals)
        is_usable_constraint_dual(row_dual) || continue
        term = constraint_certificate_term!(coefficients, constraint, row_dual)
        term === nothing && continue
        certificate += term
    end
    return certificate
end

function resolve_row_duals(constraints, batched_dual_values, scalar_dual_value)
    if batched_dual_values !== nothing
        row_duals = constraint_duals_or_nothing(constraints, batched_dual_values)
        row_duals !== nothing && return row_duals
    end
    return [constraint_dual_or_nothing(c, scalar_dual_value) for c in constraints]
end

"""
    certified_lp_bound(
        model,
        bound_type,
        objective,
        interval_bound;
        dual_value = nothing,
        dual_values = nothing,
    )

Return an LP bound certified from row duals and the declared variable bounds.

The row duals are treated as candidate Lagrange multipliers. Their signs are projected onto the
dual cones, and any stationarity residual is minimized over the variables' interval bounds. All
certificate arithmetic is outward-rounded. Unsupported constraints and unavailable duals use a
zero multiplier. If the certificate is unbounded or unavailable, return `interval_bound`.

Row duals are read per homogeneous constraint group in one batch (`dual_values`, defaulting to
a vectorized `MathOptInterface` read); a group whose batch read fails or returns an unusable
value falls back to per-constraint reads (`dual_value`, defaulting to `JuMP.dual`). Passing a
custom `dual_value` disables batch reads so every dual comes from that callback.
"""
function certified_lp_bound(
    model::JuMP.Model,
    bound_type::BoundType,
    objective::JuMPLinearType,
    interval_bound::Real;
    dual_value = nothing,
    dual_values = nothing,
)::Real
    coefficients = Dict{JuMP.VariableRef,typeof(ZERO_INTERVAL)}()
    objective_affine = convert(JuMP.AffExpr, objective)
    objective_multiplier = bound_type == lower_bound_type ? 1.0 : -1.0
    certificate =
        IntervalArithmetic.interval(objective_multiplier) *
        IntervalArithmetic.interval(objective_affine.constant)
    for (variable, coefficient) in objective_affine.terms
        add_interval_coefficient!(
            coefficients,
            variable,
            IntervalArithmetic.interval(objective_multiplier) *
            IntervalArithmetic.interval(coefficient),
        )
    end

    scalar_dual_value = dual_value === nothing ? JuMP.dual : dual_value
    batched_dual_values = if dual_value !== nothing
        nothing
    elseif dual_values !== nothing
        dual_values
    else
        constraints -> default_constraint_duals(model, constraints)
    end

    for (function_type, set_type) in JuMP.list_of_constraint_types(model)
        function_type == JuMP.AffExpr || continue
        constraints = JuMP.all_constraints(model, function_type, set_type)
        row_duals = resolve_row_duals(constraints, batched_dual_values, scalar_dual_value)
        certificate =
            add_constraint_duals_to_certificate!(coefficients, certificate, constraints, row_duals)
    end

    for (variable, coefficient) in coefficients
        IntervalArithmetic.isthinzero(coefficient) && continue
        variable_interval = variable_interval_or_nothing(variable)
        if variable_interval === nothing
            Memento.debug(
                MIPVerify.LOGGER,
                "Using interval-arithmetic bound: $(variable) has an invalid declared interval.",
            )
            return interval_bound
        end
        term = coefficient * variable_interval
        if !isfinite(lower_bound(term))
            Memento.debug(
                MIPVerify.LOGGER,
                "Using interval-arithmetic bound: $(variable) has a nonzero stationarity " *
                "residual but no finite declared bound to absorb it.",
            )
            return interval_bound
        end
        certificate += term
    end
    transformed_lower = lower_bound(certificate)
    if !isfinite(transformed_lower)
        Memento.debug(
            MIPVerify.LOGGER,
            "Using interval-arithmetic bound: the certificate value is not finite.",
        )
        return interval_bound
    end
    candidate = bound_type == lower_bound_type ? transformed_lower : -transformed_lower
    if bound_type == lower_bound_type
        return max(interval_bound, candidate)
    end
    return min(interval_bound, candidate)
end
