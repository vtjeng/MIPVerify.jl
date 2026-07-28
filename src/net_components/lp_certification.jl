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

# The MOI errors that signal an optional attribute is unavailable, as opposed to a read
# failing unexpectedly.
const UNAVAILABLE_ATTRIBUTE_ERRORS = Union{
    MathOptInterface.ResultIndexBoundsError,
    MathOptInterface.UnsupportedAttribute,
    MathOptInterface.GetAttributeNotAllowed,
}

"""
    solver_read_or_nothing(f, log_unexpected, description, consequence)

Run `f` and return its result, or `nothing` if the read fails.

Reading an optional solver attribute (a row dual, an objective bound) can tighten a bound but
is never required for soundness, so the caller always has a valid fallback.
`UNAVAILABLE_ATTRIBUTE_ERRORS` are expected and return `nothing` quietly. Any other error is
logged via `log_unexpected` and also returns `nothing`, so one failed read degrades a single
bound instead of crashing the run; interrupts and resource exhaustion still propagate.
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
        if !(error isa UNAVAILABLE_ATTRIBUTE_ERRORS)
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

"""
    single_constraint_dual_or_nothing(dual_values, index)

Read one row's dual through the batched `dual_values` source as a single-element batch,
returning `nothing` when the read fails or the result is not a one-element vector.
"""
function single_constraint_dual_or_nothing(dual_values, index)
    values = solver_read_or_nothing(
        () -> dual_values([index]),
        Memento.warn,
        "a constraint dual",
        "treating it as unavailable",
    )
    values === nothing && return nothing
    if !(values isa AbstractVector) || length(values) != 1
        Memento.warn(
            MIPVerify.LOGGER,
            "Single constraint-dual retry returned an incompatible value; " *
            "treating it as unavailable.",
        )
        return nothing
    end
    return first(values)
end

function default_constraint_duals(model::JuMP.Model, indices)
    JuMP.has_duals(model) || return nothing
    return MathOptInterface.get(JuMP.backend(model), MathOptInterface.ConstraintDual(), indices)
end

function constraint_duals_or_nothing(indices, dual_values)
    values = solver_read_or_nothing(
        () -> dual_values(indices),
        Memento.debug,
        "a batch of constraint duals",
        "retrying individually",
    )
    values === nothing && return nothing
    if !(values isa AbstractVector) || length(values) != length(indices)
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
    constraint_certificate_term!(model, coefficients, index, row_dual)

Apply one row's certificate contribution, split in two: the row's variable terms are subtracted
into `coefficients` (mutated in place), and its scalar `multiplier * (reference - constant)`
term is returned. Returns `nothing`, leaving `coefficients` unchanged, when the row's set is
unsupported or the projected multiplier or reference is unusable.

`index` is the row's `MOI.ConstraintIndex`. Reading the set and the function through the index
is what `JuMP.constraint_object` does, without its conversion of the function into an `AffExpr`.
Only a row carrying a usable dual reaches this function, so most rows are never read at all.

The `{F,S}` parameters carry the row's function and set types, and the two reads are annotated
with them. Keep both annotations. `JuMP.Model` stores its backend in a field typed
`MathOptInterface.ModelLike`, so `JuMP.backend(model)` is abstract here and each
`MathOptInterface.get` call sees an abstract first argument. Strip the annotations and both
reads infer as `Any`, which dispatches the set projection and every term dynamically. Checking
the same two `get` calls against a concretely typed backend infers `F` and `S`, so that check
reports the annotations as redundant and does not reproduce the `Any` seen here.
"""
function constraint_certificate_term!(
    model::JuMP.Model,
    coefficients,
    index::MathOptInterface.ConstraintIndex{F,S},
    row_dual::Real,
) where {F,S}
    backend = JuMP.backend(model)
    set = MathOptInterface.get(backend, MathOptInterface.ConstraintSet(), index)::S
    projected = projected_dual_and_reference(set, row_dual)
    projected === nothing && return nothing
    multiplier, reference = projected
    (iszero(multiplier) || !isfinite(reference)) && return nothing
    multiplier_interval = IntervalArithmetic.interval(multiplier)
    row = MathOptInterface.get(backend, MathOptInterface.ConstraintFunction(), index)::F
    for term in row.terms
        add_interval_coefficient!(
            coefficients,
            JuMP.VariableRef(model, term.variable),
            -multiplier_interval * IntervalArithmetic.interval(term.coefficient),
        )
    end
    return multiplier_interval *
           (IntervalArithmetic.interval(reference) - IntervalArithmetic.interval(row.constant))
end

function add_constraint_duals_to_certificate!(
    model::JuMP.Model,
    coefficients,
    certificate,
    indices,
    row_duals,
)
    for (index, row_dual) in zip(indices, row_duals)
        is_usable_constraint_dual(row_dual) || continue
        term = constraint_certificate_term!(model, coefficients, index, row_dual)
        term === nothing && continue
        certificate += term
    end
    return certificate
end

function resolve_row_duals(indices, dual_values)
    row_duals = constraint_duals_or_nothing(indices, dual_values)
    row_duals !== nothing && return row_duals
    return [single_constraint_dual_or_nothing(dual_values, i) for i in indices]
end

# The MOI function type carried by `AffExpr` constraints, `ScalarAffineFunction{Float64}`.
const AFFINE_MOI_FUNCTION_TYPE = JuMP.moi_function_type(JuMP.AffExpr)

"""
    affine_constraint_indices(model, set_type)

Return the `MOI.ConstraintIndex` of every `AffExpr`-in-`set_type` row of `model`.

The function type is fixed to `AFFINE_MOI_FUNCTION_TYPE`, so this enumerates only the affine
rows of `set_type`. `certified_lp_bound` restricts its loop to the same rows through a
`function_type == JuMP.AffExpr` guard, so the two agree on which rows the certificate covers.

`certified_lp_bound` runs once per LP bound solve, and a single sample can need hundreds of them.
Enumerating the indices is the cheap half of `JuMP.all_constraints`, which spends almost all of
its time wrapping each index in a `JuMP.ConstraintRef`. The certificate reads each row's dual,
set, and function straight off the index, so it never needs that wrapper.
"""
function affine_constraint_indices(model::JuMP.Model, set_type::DataType)
    return MathOptInterface.get(
        model,
        MathOptInterface.ListOfConstraintIndices{AFFINE_MOI_FUNCTION_TYPE,set_type}(),
    )
end

"""
    certified_lp_bound(model, bound_type, objective, interval_bound)

Return an LP bound certified from row duals and the declared variable bounds.

The row duals are treated as candidate Lagrange multipliers. Their signs are projected onto the
dual cones, and any stationarity residual is minimized over the variables' interval bounds. All
certificate arithmetic is outward-rounded. Unsupported constraints and unavailable duals use a
zero multiplier. If the certificate is unbounded or unavailable, return `interval_bound`.

Every dual is read from the model's backend through the vectorized `MathOptInterface`
interface: each homogeneous constraint group in one batch and, when a group read fails or
returns a value of the wrong shape, each constraint retried as a single-element batch.
Individual elements of a well-shaped batch that are not finite nonzero reals are skipped
without a retry.
"""
function certified_lp_bound(
    model::JuMP.Model,
    bound_type::BoundType,
    objective::JuMPLinearType,
    interval_bound::Real,
)::Real
    dual_values = constraints -> default_constraint_duals(model, constraints)
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

    for (function_type, set_type) in JuMP.list_of_constraint_types(model)
        function_type == JuMP.AffExpr || continue
        indices = affine_constraint_indices(model, set_type)
        row_duals = resolve_row_duals(indices, dual_values)
        certificate = add_constraint_duals_to_certificate!(
            model,
            coefficients,
            certificate,
            indices,
            row_duals,
        )
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
