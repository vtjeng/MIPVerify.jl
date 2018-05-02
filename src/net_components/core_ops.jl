using JuMP
using ConditionalJuMP
using Memento

function is_constant(x::JuMP.AffExpr)
    x.vars |> length == 0
end

function is_constant(x::JuMP.Variable)
    false
end

function get_tightening_algorithm(
    x::JuMPLinearType,
    nta::Nullable{TighteningAlgorithm})::TighteningAlgorithm
    if !isnull(nta)
        return get(nta)
    else
        if is_constant(x)
            return interval_arithmetic
        end
        m = ConditionalJuMP.getmodel(x)
        return !haskey(m.ext, :MIPVerify) ? DEFAULT_TIGHTENING_ALGORITHM : m.ext[:MIPVerify].tightening_algorithm
    end
end

@enum BoundType lower_bound_type=-1 upper_bound_type=1
bound_f = Dict(
    lower_bound_type => lowerbound,
    upper_bound_type => upperbound
)
bound_obj = Dict(
    lower_bound_type => :Min,
    upper_bound_type => :Max
)
bound_delta_f = Dict(
    lower_bound_type => (b, b_0) -> b - b_0,
    upper_bound_type => (b, b_0) -> b_0 - b
)

function tight_bound(
    x::JuMPLinearType, 
    nta::Nullable{TighteningAlgorithm},
    bound_type::BoundType)
    tightening_algorithm = get_tightening_algorithm(x, nta)
    b_0 = bound_f[bound_type](x)
    if tightening_algorithm == interval_arithmetic || is_constant(x)
        return b_0
    end
    relaxation = (tightening_algorithm == lp)
    m = ConditionalJuMP.getmodel(x)
    @objective(m, bound_obj[bound_type], x)
    status = solve(m, suppress_warnings = true, relaxation=relaxation)
    if status == :Optimal
        b = getobjectivevalue(m)
    elseif status == :UserLimit
        b = getobjectivebound(m)
        log_gap(m)
    else
        warn(MIPVerify.LOGGER, "Unexpected solve status $(status) while tightening via $(tightening_algorithm); using interval_arithmetic to obtain upperbound.")
        b = b_0
    end
    db = bound_delta_f[bound_type](b, b_0)
    debug(MIPVerify.LOGGER, "  Δu = $(db)")
    if db < 0
        b = b_0
        info(MIPVerify.LOGGER, "Tightening via interval_arithmetic gives a better result than $(tightening_algorithm); using best bound found.")
    end
    return b
end

function tight_upperbound(
    x::JuMPLinearType; 
    nta::Nullable{TighteningAlgorithm} = Nullable{TighteningAlgorithm}())
    tight_bound(x, nta, upper_bound_type)
end

function tight_lowerbound(
    x::JuMPLinearType;
    nta::Nullable{TighteningAlgorithm} = Nullable{TighteningAlgorithm}())
    tight_bound(x, nta, lower_bound_type)
end

function log_gap(m::JuMP.Model)
    gap = abs(1-getobjectivebound(m)/getobjectivevalue(m))
    info(MIPVerify.LOGGER, "Hit user limit during solve to determine bounds. Multiplicative gap was $gap.")
end

function relu(x::T)::T where {T<:Real}
    return max(zero(T), x)
end

function relu(x::AbstractArray{T}) where {T<:Real}
    return relu.(x)
end

function relu(x::T, l::Real, u::Real)::JuMP.AffExpr where {T<:JuMPLinearType}
    M = max(u, l, -u, -l)
    
    model = ConditionalJuMP.getmodel(x)
    x_rect = @variable(model)
    a = @variable(model, category = :Bin)
    
    @constraint(model, x_rect <= x + M*(1-a))
    @constraint(model, x_rect >= x)
    @constraint(model, x_rect <= M*a)
    @constraint(model, x_rect >= 0)

    # Manually set the bounds for x_rect so they can be used by downstream operations.
    setlowerbound(x_rect, 0)
    setupperbound(x_rect, M)
    return x_rect
end

@enum ReLUType split=0 zero_output=-1 linear_in_input=1 constant_output=2

function get_relu_type(l::Real, u::Real)::ReLUType
    if u <= 0
        return zero_output
    elseif u==l
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
        n = count(x -> x==t, relutypes)
        print(io, "$t: $n")
        if t != last(instances(ReLUType))
            print(io, ", ")
        end
    end
end

function relu(x::JuMPLinearType)::JuMP.AffExpr
    u = tight_upperbound(x)
    l = tight_lowerbound(x)
    relu(x, l, u)
end

"""
$(SIGNATURES)
Expresses a rectified-linearity constraint: output is constrained to be equal to 
`max(x, 0)`.
"""
function relu(
    x::AbstractArray{T}; 
    nta::Nullable{TighteningAlgorithm} = Nullable{TighteningAlgorithm}())::Array{JuMP.AffExpr} where {T<:JuMPLinearType}
    show_progress_bar::Bool = MIPVerify.LOGGER.levels[MIPVerify.LOGGER.level] > MIPVerify.LOGGER.levels["debug"]
    if !show_progress_bar
        u = tight_upperbound.(x, nta=nta)
        l = tight_lowerbound.(x, nta=nta)
        return relu.(x, l, u)
    else
        p1 = Progress(length(x), desc="  Calculating upper bounds: ")
        u = map(x_i -> (next!(p1); tight_upperbound(x_i, nta=nta)), x)
        p2 = Progress(length(x), desc="  Calculating lower bounds: ")
        l = map(x_i -> (next!(p2); tight_lowerbound(x_i, nta=nta)), x)

        reluinfo = ReLUInfo(l, u)
        info(MIPVerify.LOGGER, "$reluinfo")

        p3 = Progress(length(x), desc="  Imposing relu constraint: ")
        return x_r = map(v -> (next!(p3); relu(v...)), zip(x, l, u))
    end
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
    nta::Nullable{TighteningAlgorithm} = Nullable{TighteningAlgorithm}())::Array{JuMP.AffExpr}
    @assert(size(x) == size(m))
    s = size(m)
    # We add the constraints corresponding to the active ReLUs to the model
    zero_idx = Iterators.filter(i -> m[i]==0, CartesianRange(s)) |> collect
    d = Dict(zip(zero_idx, relu(x[zero_idx], nta=nta)))

    # We determine the output of the masked relu, which is either: 
    #  1) the output of the relu that we have previously determined when adding the 
    #     constraints to the model. 
    #  2, 3) the result of applying the (elementwise) masked_relu function.
    return map(i -> m[i] == 0 ? d[i] : masked_relu(x[i], m[i]), CartesianRange(s))
end

function maximum(xs::AbstractArray{T})::T where {T<:Real}
    return Base.maximum(xs)
end

"""
$(SIGNATURES)
Expresses a maximization constraint: output is constrained to be equal to `max(xs)`.
"""
function maximum(xs::AbstractArray{T})::JuMP.AffExpr where {T<:JuMPLinearType}
    if length(xs) == 1
        return xs[1]
    end

    model = ConditionalJuMP.getmodel(xs[1])

    # TODO (vtjeng): [PERF] skip calculating lowerbound for index if upperbound is lower than
    # largest current lowerbound.
    p1 = Progress(length(xs), desc="  Calculating upper bounds: ")
    us = map(x_i -> (next!(p1); tight_upperbound(x_i)), xs)
    p2 = Progress(length(xs), desc="  Calculating lower bounds: ")
    ls = map(x_i -> (next!(p2); tight_lowerbound(x_i)), xs)

    l = Base.maximum(ls)
    u = Base.maximum(us)

    if l==u
        return one(T)*l
        info(MIPVerify.LOGGER, "Output of maximum is constant.")
    end
    # at least one index will satisfy this property because of check above.
    filtered_indexes = us .> l
    
    # TODO (vtjeng): Smarter debugging if maximum is being used more than once.
    info(MIPVerify.LOGGER, "Number of inputs to maximum function possibly taking maximum value: $(filtered_indexes |> sum)")
    
    return maximum(xs[filtered_indexes], ls[filtered_indexes], us[filtered_indexes])
end

function maximum(
    xs::AbstractArray{T, 1},
    ls::AbstractArray{<:Real, 1},
    us::AbstractArray{<:Real, 1},
    )::JuMP.AffExpr where {T<:JuMPLinearType}

    @assert length(xs)>0
    @assert length(xs)==length(ls)
    @assert length(xs)==length(us)

    model = ConditionalJuMP.getmodel(xs[1])
    if length(xs) == 1
        return first(xs)
    else
        l = Base.maximum(ls)
        u = Base.maximum(us)
        x_max = @variable(model, lowerbound = l, upperbound=u)
        a = @variable(model, [1:length(xs)], category =:Bin)
        @constraint(model, sum(a) == 1)
        for (i, x) in enumerate(xs)
            umaxi = Base.maximum(us[1:end .!= i])
            @constraint(model, x_max <= x + (1-a[i])*(umaxi - ls[i]))
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
"""
function maximum_ge(xs::AbstractArray{T})::JuMP.Variable where {T<:JuMPLinearType}
    @assert length(xs)>0
    model = ConditionalJuMP.getmodel(xs[1])
    x_max = @variable(model)
    @constraint(model, x_max .>= xs)
    return x_max
end

"""
$(SIGNATURES)
Expresses a one-sided absolute-value constraint: output is constrained to be at least as
large as `|x|`.

Only use when you are minimizing over the output in the objective.
"""
function abs_ge(x::JuMPLinearType)::JuMP.AffExpr
    model = ConditionalJuMP.getmodel(x)
    u = upperbound(x)
    l = lowerbound(x)
    if u <= 0
        return -x
    elseif l >= 0
        return x
    else
        x_abs = @variable(model)
        @constraint(model, x_abs >= x)
        @constraint(model, x_abs >= -x)
        setlowerbound(x_abs, 0)
        setupperbound(x_abs, max(-l, u))
        return x_abs
    end
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
    x::Array{<:JuMPLinearType, 1},
    target_indexes::Array{<:Integer, 1};
    tolerance::Real = 0)
    
    @assert length(x) >= 1
    model = ConditionalJuMP.getmodel(x[1])

    target_vars = x[Bool[i∈target_indexes for i = 1:length(x)]]
    other_vars = x[Bool[i∉target_indexes for i = 1:length(x)]]

    maximum_target_var = length(target_vars) == 1 ?
        target_vars[1] :    
        MIPVerify.maximum(target_vars)

    @constraint(model, other_vars - maximum_target_var .<= -tolerance)
end