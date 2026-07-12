function add_interval_coefficient!(coefficients, variable::JuMP.VariableRef, delta)
    zero_interval = IntervalArithmetic.interval(0.0)
    coefficients[variable] = get(coefficients, variable, zero_interval) + delta
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

function constraint_dual_or_nothing(constraint, dual_value)
    value = try
        dual_value(constraint)
    catch error
        if error isa MathOptInterface.ResultIndexBoundsError ||
           error isa MathOptInterface.UnsupportedAttribute ||
           error isa MathOptInterface.GetAttributeNotAllowed
            return nothing
        end
        rethrow()
    end
    return value isa Real && isfinite(value) ? value : nothing
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

"""
    certified_lp_bound(model, bound_type, objective, interval_bound; dual_value = JuMP.dual)

Return an LP bound certified from row duals and the declared variable bounds.

The row duals are treated as candidate Lagrange multipliers. Their signs are projected onto the
dual cones, and any stationarity residual is minimized over the variables' interval bounds. All
certificate arithmetic is outward-rounded. Unsupported constraints and unavailable duals use a
zero multiplier. If the certificate is unbounded or unavailable, return `interval_bound`.
"""
function certified_lp_bound(
    model::JuMP.Model,
    bound_type::BoundType,
    objective::JuMPLinearType,
    interval_bound::Real;
    dual_value = JuMP.dual,
)::Real
    zero_interval = IntervalArithmetic.interval(0.0)
    interval_type = typeof(zero_interval)
    coefficients = Dict{JuMP.VariableRef,interval_type}()
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
        for constraint in JuMP.all_constraints(model, function_type, set_type)
            row_dual = constraint_dual_or_nothing(constraint, dual_value)
            row_dual === nothing && continue
            iszero(row_dual) && continue
            constraint_object = JuMP.constraint_object(constraint)
            projected = projected_dual_and_reference(constraint_object.set, row_dual)
            projected === nothing && continue
            multiplier, reference = projected
            (iszero(multiplier) || !isfinite(reference)) && continue
            multiplier_interval = IntervalArithmetic.interval(multiplier)
            function_value = constraint_object.func
            certificate +=
                multiplier_interval * (
                    IntervalArithmetic.interval(reference) -
                    IntervalArithmetic.interval(function_value.constant)
                )
            for (variable, coefficient) in function_value.terms
                add_interval_coefficient!(
                    coefficients,
                    variable,
                    -multiplier_interval * IntervalArithmetic.interval(coefficient),
                )
            end
        end
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
