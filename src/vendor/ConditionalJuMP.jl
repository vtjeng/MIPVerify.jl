using JuMP
using IntervalArithmetic: Interval

# We vendor https://github.com/rdeits/ConditionalJuMP.jl/blob/e0c406077c0b07be76e02f72c3a7a7aa650df82f/src/ConditionalJuMP.jl
# so that we can use JuMP >= 0.2.0

# https://github.com/rdeits/ConditionalJuMP.jl/blob/e0c406077c0b07be76e02f72c3a7a7aa650df82f/src/ConditionalJuMP.jl#L430-L431
getmodel(x::JuMP.VariableRef) = x.m
getmodel(x::JuMP.GenericAffExpr) = first(x.vars).m

function getmodel(xs::AbstractArray{T}) where {T<:Union{JuMP.VariableRef,JuMP.GenericAffExpr}}
    for x in xs
        if !is_constant(x)
            return getmodel(x)
        end
    end
    throw(DomainError("Array contains only constants, so no model can be determine"))
end

interval(x::Number) = Interval(x, x)
interval(x::JuMP.VariableRef) = Interval(JuMP.getlowerbound(x), JuMP.getupperbound(x))
function interval(e::JuMP.GenericAffExpr)
    if isempty(e.coeffs)
        return Interval(e.constant, e.constant)
    else
        result = Interval(e.constant, e.constant)
        for i in eachindex(e.coeffs)
            var = e.vars[i]
            coef = e.coeffs[i]
            result += Interval(coef, coef) * Interval(getlowerbound(var), getupperbound(var))
        end
        return result
    end
end

lowerbound(x::Number) = x
upperbound(x::Number) = x
lowerbound(x::JuMP.VariableRef) = JuMP.getlowerbound(x)
upperbound(x::JuMP.VariableRef) = JuMP.getupperbound(x)
lowerbound(e::JuMP.GenericAffExpr) = lowerbound(interval(e))
upperbound(e::JuMP.GenericAffExpr) = upperbound(interval(e))
lowerbound(i::Interval) = i.lo
upperbound(i::Interval) = i.hi
