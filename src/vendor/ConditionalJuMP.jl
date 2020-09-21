using JuMP
using IntervalArithmetic: Interval

# We vendor ConditionalJuMP (https://github.com/rdeits/ConditionalJuMP.jl/blob/e0c406077c0b07be76e02f72c3a7a7aa650df82f/src/ConditionalJuMP.jl)
# so that we can use JuMP >= 0.2.0

owner_model(x::JuMP.VariableRef)::Model = JuMP.owner_model(x)
function owner_model(x::JuMP.GenericAffExpr)::Union{Model,Nothing}
    if length(x.terms) == 0
        return nothing
    end
    return JuMP.owner_model(first(x.terms.keys))
end
function owner_model(
    xs::AbstractArray{T},
)::Model where {T<:Union{JuMP.VariableRef,JuMP.GenericAffExpr}}
    for x in xs
        m = owner_model(x)
        if m !== nothing
            return m
        end
    end
    return nothing
end

interval(x::Number) = Interval(x, x)
interval(x::JuMP.VariableRef) = Interval(lower_bound(x), upper_bound(x))
function interval(e::JuMP.GenericAffExpr)
    result = Interval(e.constant, e.constant)
    for (var, coef) in e.terms
        result += Interval(coef, coef) * Interval(lower_bound(var), upper_bound(var))
    end
    return result
end

lower_bound(x::Number) = x
upper_bound(x::Number) = x
lower_bound(x::JuMP.VariableRef) = JuMP.lower_bound(x)
upper_bound(x::JuMP.VariableRef) = JuMP.upper_bound(x)
lower_bound(e::JuMP.GenericAffExpr) = lower_bound(interval(e))
upper_bound(e::JuMP.GenericAffExpr) = upper_bound(interval(e))
lower_bound(i::Interval) = i.lo
upper_bound(i::Interval) = i.hi
