using JuMP
using ConditionalJuMP
using Memento

function do_tighten_bounds(
    x::JuMP.AbstractJuMPScalar)::Bool
    m = ConditionalJuMP.getmodel(x)
    if !haskey(m.ext, :MIPVerify)
        # always tighten bounds if unspecified
        return true
    else
        return m.ext[:MIPVerify].tighten_bounds
    end
end

function tight_upperbound(
    x::JuMP.AbstractJuMPScalar; 
    tighten::Bool = do_tighten_bounds(x))
    u = upperbound(x)
    if !tighten
        return u
    end
    m = ConditionalJuMP.getmodel(x)
    @objective(m, Max, x)
    status = solve(m, suppress_warnings = true)
    if status == :Optimal || status == :UserLimit
        u = min(getobjectivebound(m), u)
        if status == :UserLimit
            log_gap(m)
        end
    end
    debug(MIPVerify.LOGGER, "  Δu = $(upperbound(x)-u)")
    return u
end

function tight_lowerbound(
    x::JuMP.AbstractJuMPScalar;
    tighten::Bool = do_tighten_bounds(x))
    l = lowerbound(x)
    if !tighten
        return l
    end
    m = ConditionalJuMP.getmodel(x)
    @objective(m, Min, x)
    status = solve(m, suppress_warnings = true)
    if status == :Optimal || status == :UserLimit
        l = max(getobjectivebound(m), l)
        if status == :UserLimit
            log_gap(m)
        end
    end
    debug(MIPVerify.LOGGER, "  Δl = $(l-lowerbound(x))")
    return l
end

function log_gap(m::JuMP.Model)
    gap = abs(1-getobjectivebound(m)/getobjectivevalue(m))
    info(MIPVerify.LOGGER, "Hit user limit during solve to determine bounds. Multiplicative gap was $gap.")
end

function relu(x::Real)::Real
    return max(0, x)
end

"""
$(SIGNATURES)
Expresses a rectified-linearity constraint: output is constrained to be equal to 
`max(x, 0)`.
"""
function relu(x::JuMP.AbstractJuMPScalar)::JuMP.AbstractJuMPScalar
    model = ConditionalJuMP.getmodel(x)
    x_rect = @variable(model)
    u = tight_upperbound(x)
    l = tight_lowerbound(x)

    if u <= 0
        # rectified value is always 0
        @constraint(model, x_rect == 0)
        setlowerbound(x_rect, 0)
        setupperbound(x_rect, 0)
    elseif l >= 0
        # rectified value is always equal to x itself.
        @constraint(model, x_rect == x)
        setlowerbound(x_rect, l)
        setupperbound(x_rect, u)
    else
        a = @variable(model, category = :Bin)

        # refined big-M formulation that takes advantage of the knowledge
        # that lower and upper bounds  are different.
        @constraint(model, x_rect <= x + (-l)*(1-a))
        @constraint(model, x_rect >= x)
        @constraint(model, x_rect <= u*a)
        @constraint(model, x_rect >= 0)

        # model.ext[:objective] = get(model.ext, :objective, 0) + x_rect - x
        model.ext[:objective] = get(model.ext, :objective, 0) + x_rect - x*u/(u-l)

        # Manually set the bounds for x_rect so they can be used by downstream operations.
        setlowerbound(x_rect, 0)
        setupperbound(x_rect, u)
    end

    return x_rect
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
function masked_relu(x::JuMP.AbstractJuMPScalar, m::Real)::JuMP.Variable
    if m < 0
        # TODO (vtjeng): this is bad! we should be able to completely remove this
        # extraneous variable.
        model = ConditionalJuMP.getmodel(x)
        x_0 = @variable(model)
        JuMP.fix(x_0, 0)
        return x_0
    elseif m > 0
        model = ConditionalJuMP.getmodel(x)
        x_id = @variable(model)
        @constraint(model, x_id == x)
        setupperbound(x_id, upperbound(x))
        setlowerbound(x_id, lowerbound(x))
        return x_id
    else
        return relu(x)
    end
end

function masked_relu(x::Real, m::Real)::Real
    if m < 0
        0
    elseif m > 0
        x
    else
        relu(x)
    end
end

function maximum(xs::AbstractArray{T})::T where {T<:Real}
    return Base.maximum(xs)
end

"""
$(SIGNATURES)
Expresses a maximization constraint: output is constrained to be equal to `max(xs)`.
"""
function maximum(
    xs::AbstractArray{T}; 
    tighten::Bool = do_tighten_bounds(xs[1]))::JuMP.Variable where {T<:JuMP.AbstractJuMPScalar}
    model = ConditionalJuMP.getmodel(xs[1])
    ls = tight_lowerbound.(xs; tighten = tighten)
    us = tight_upperbound.(xs; tighten = tighten)
    l = Base.maximum(ls)
    u = Base.maximum(us)
    x_max = @variable(model,
        lowerbound = l,
        upperbound = u)
    
    xs_filtered::Array{T, 1} = map(
        t-> t[1], 
        Iterators.filter(
            t -> t[2]>l, 
            zip(xs, us)
        )
    )

    if length(xs_filtered) == 1
        @constraint(model, x_max == xs_filtered[1])
    else
        indicators = []
        for (i, x) in enumerate(xs_filtered)
            a = @variable(model, category =:Bin)
            umaxi = Base.maximum(us[1:end .!= i])
            @constraint(model, x_max <= x + (1-a)*(umaxi - ls[i]))
            @constraint(model, x_max >= x)
            push!(indicators, a)
        end
        @constraint(model, sum(indicators) == 1)
    end
    return x_max
end

"""
$(SIGNATURES)
Expresses a one-sided absolute-value constraint: output is constrained to be at least as
large as `|x|`.

Only use when you are minimizing over the output in the objective.
"""
function abs_ge(x::JuMP.AbstractJuMPScalar)::JuMP.Variable
    model = ConditionalJuMP.getmodel(x)
    x_abs = @variable(model)
    u = upperbound(x)
    l = lowerbound(x)
    if u <= 0
        @constraint(model, x_abs == -x)
        setlowerbound(x_abs, -u)
        setupperbound(x_abs, -l)
    elseif l >= 0
        @constraint(model, x_abs == x)
        setlowerbound(x_abs, l)
        setupperbound(x_abs, u)
    else
        @constraint(model, x_abs >= x)
        @constraint(model, x_abs >= -x)
        setlowerbound(x_abs, 0)
        setupperbound(x_abs, max(-l, u))
    end
    return x_abs
end

function get_target_indexes(
    target_index::Integer,
    array_length::Integer;
    invert_target_selection::Bool = false)
    
    get_target_indexes([target_index], array_length, invert_target_selection = invert_target_selection)

end

function get_target_indexes(
    target_indexes::Array{<:Integer, 1},
    array_length::Integer;
    invert_target_selection::Bool = false)

    @assert length(target_indexes) >= 1
    @assert all(target_indexes .>= 1) && all(target_indexes .<= array_length)
    
    invert_target_selection ?
        filter((x) -> x ∉ target_indexes, 1:array_length) :
        target_indexes
end

"""
$(SIGNATURES)

Imposes constraints ensuring that one of the elements at the target_indexes is the 
largest element of the array x. More specifically, we require `x[j] - x[i] ≥ tolerance` for
some `j ∈ target_indexes` and for all `i ∉ target_indexes`.
"""
function set_max_indexes(
    x::Array{<:JuMP.AbstractJuMPScalar, 1},
    target_indexes::Array{<:Integer, 1};
    tolerance::Real = 0)
    
    @assert length(x) >= 1
    model = ConditionalJuMP.getmodel(x[1])

    target_vars = x[Bool[i∈target_indexes for i = 1:length(x)]]
    other_vars = x[Bool[i∉target_indexes for i = 1:length(x)]]

    maximum_target_var = length(target_vars) == 1 ?
        target_vars[1] :    
        MIPVerify.maximum(target_vars; tighten = false)

    @constraint(model, other_vars - maximum_target_var .<= -tolerance)
end