using JuMP
using ConditionalJuMP
using Memento

"""
$(SIGNATURES)

Checks whether a JuMPLinearType is constant (and thus has no model associated)
with it. This can only be true if it is an affine expression with no stored
variables.
"""
function is_constant(x::JuMP.AffExpr)
    x.vars |> length == 0
end

function is_constant(x::JuMP.Variable)
    false
end

function getmodel(xs::AbstractArray{T}) where {T<:JuMPLinearType}
    for x in xs
        if !is_constant(x)
            return ConditionalJuMP.getmodel(x)
        end
    end
    throw(DomainError("None of the JuMPLinearTypes has an associated model."))
end

function get_tightening_algorithm(
    x::JuMPLinearType,
    nta::Nullable{TighteningAlgorithm})::TighteningAlgorithm
    if is_constant(x)
        return interval_arithmetic
    elseif !isnull(nta)
        return get(nta)
    else
        # x is not constant, and thus x must have an associated model
        model = ConditionalJuMP.getmodel(x)
        return !haskey(model.ext, :MIPVerify) ? DEFAULT_TIGHTENING_ALGORITHM : model.ext[:MIPVerify].tightening_algorithm
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
bound_operator = Dict(
    lower_bound_type => >=,
    upper_bound_type => <=
)

"""
Calculates a tight bound of type `bound_type` on the variable `x` using the specified 
tightening algorithm `nta`.

If an upper bound is proven to be below cutoff, or a lower bound is proven to above cutoff,
the algorithm returns early with whatever value was found.
"""
function tight_bound(
    x::JuMPLinearType, 
    nta::Nullable{TighteningAlgorithm},
    bound_type::BoundType,
    cutoff::Real)
    tightening_algorithm = get_tightening_algorithm(x, nta)
    b_0 = bound_f[bound_type](x)
    if tightening_algorithm == interval_arithmetic || is_constant(x) || bound_operator[bound_type](b_0, cutoff)
        return b_0
    end
    relaxation = (tightening_algorithm == lp)
    # x is not constant, and thus x must have an associated model
    model = ConditionalJuMP.getmodel(x)
    @objective(model, bound_obj[bound_type], x)
    status = solve(model, suppress_warnings = true, relaxation=relaxation)
    if status == :Optimal
        b = getobjectivevalue(model)
    elseif status == :UserLimit
        b = getobjectivebound(model)
        log_gap(model)
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
    nta::Nullable{TighteningAlgorithm} = Nullable{TighteningAlgorithm}(),
    cutoff::Real = -Inf)
    tight_bound(x, nta, upper_bound_type, cutoff)
end

function tight_lowerbound(
    x::JuMPLinearType;
    nta::Nullable{TighteningAlgorithm} = Nullable{TighteningAlgorithm}(),
    cutoff::Real = Inf)
    tight_bound(x, nta, lower_bound_type, cutoff)
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
    if u<l
        # TODO (vtjeng): This check is in place in case of numerical error in the calculation of bounds. 
        # See sample number 4872 (1-indexed) when verified on the lp0.4 network.
        warn(MIPVerify.LOGGER, "Inconsistent upper and lower bounds: u-l = $(u-l) is negative. Attempting to use interval arithmetic bounds instead ...")
        u=upperbound(x)
        l=lowerbound(x)
    end

    if u <= 0
        # rectified value is always 0
        return zero(T)
    elseif u==l
        return one(T)*l
    elseif u<l
        error(MIPVerify.LOGGER, "Inconsistent upper and lower bounds even after using only interval arithmetic: u-l = $(u-l) is negative")
    elseif l >= 0
        # rectified value is always x
        return x
    else
        # since we know that u!=l, x is not constant, and thus x must have an associated model
        model = ConditionalJuMP.getmodel(x)
        x_rect = @variable(model)
        a = @variable(model, category = :Bin)

        # refined big-M formulation that takes advantage of the knowledge
        # that lower and upper bounds  are different.
        @constraint(model, x_rect <= x + (-l)*(1-a))
        @constraint(model, x_rect >= x)
        @constraint(model, x_rect <= u*a)
        @constraint(model, x_rect >= 0)

        # Manually set the bounds for x_rect so they can be used by downstream operations.
        setlowerbound(x_rect, 0)
        setupperbound(x_rect, u)
        return x_rect
    end
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

"""
Calculates the lowerbound only if `u` is positive; otherwise, returns `u` (since we expect)
the ReLU to be fixed to zero anyway.
"""
function lazy_tight_lowerbound(
    x::JuMPLinearType, u::Real; 
    nta::Nullable{TighteningAlgorithm} = Nullable{TighteningAlgorithm}(),
    cutoff=0
    )::Real
    (u <= cutoff) ? u : tight_lowerbound(x; nta = nta, cutoff=cutoff)
end

function relu(x::JuMPLinearType)::JuMP.AffExpr
    u = tight_upperbound(x, cutoff=0)
    l = lazy_tight_lowerbound(x, u, cutoff=0)
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
        u = tight_upperbound.(x, nta=nta, cutoff=0)
        l = lazy_tight_lowerbound.(x, u, nta=nta, cutoff=0)
        return relu.(x, l, u)
    else
        p1 = Progress(length(x), desc="  Calculating upper bounds: ")
        u = map(x_i -> (next!(p1); tight_upperbound(x_i, nta=nta, cutoff=0)), x)
        p2 = Progress(length(x), desc="  Calculating lower bounds: ")
        l = map(v -> (next!(p2); lazy_tight_lowerbound(v..., nta=nta, cutoff=0)), zip(x, u))

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

function maximum_of_constants(xs::AbstractArray{T}) where {T<:JuMPLinearType}
    @assert all(is_constant.(xs))
    max_val = map(x -> x.constant, xs) |> maximum
    return one(JuMP.Variable)*max_val
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
    model = MIPVerify.getmodel(xs)

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
    
    # TODO (vtjeng): Smarter log output if maximum function is being used more than once (for example, in a max-pooling layer).
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
    
    if all(is_constant.(xs))
        return maximum_of_constants(xs)
    end
    # at least one of xs is not constant.
    model = MIPVerify.getmodel(xs)
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

NB: If all of xs are constant, we simply return the largest of them.
"""
function maximum_ge(xs::AbstractArray{T})::JuMP.Variable where {T<:JuMPLinearType}
    @assert length(xs)>0
    if all(is_constant.(xs))
        return maximum_of_constants(xs)
    end
    # at least one of xs is not constant.
    model = MIPVerify.getmodel(xs)
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
    if is_constant(x)
        return one(JuMP.Variable)*abs(x.constant)
    end
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

function get_vars_for_max_index(
    xs::Array{<:JuMPLinearType, 1},
    target_indexes::Array{<:Integer, 1},
    tolerance::Real)

    @assert length(xs) >= 1

    target_vars = xs[Bool[i∈target_indexes for i = 1:length(xs)]]
    other_vars = xs[Bool[i∉target_indexes for i = 1:length(xs)]]

    maximum_target_var = length(target_vars) == 1 ?
        target_vars[1] :    
        MIPVerify.maximum(target_vars)

    return (maximum_target_var, other_vars)
end

"""
$(SIGNATURES)

Imposes constraints ensuring that one of the elements at the target_indexes is the 
largest element of the array x. More specifically, we require `x[j] - x[i] ≥ tolerance` for
some `j ∈ target_indexes` and for all `i ∉ target_indexes`.
"""
function set_max_indexes(
    model::Model,
    xs::Array{<:JuMPLinearType, 1},
    target_indexes::Array{<:Integer, 1};
    tolerance::Real = 0)

    (maximum_target_var, other_vars) = get_vars_for_max_index(xs, target_indexes, tolerance)

    @constraint(model, other_vars - maximum_target_var .<= -tolerance)
end