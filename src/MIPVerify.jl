module MIPVerify

using Base.Cartesian
using JuMP
using ConditionalJuMP
using Memento
using AutoHashEquals

include("layers/core_ops.jl")

include("layers/net_parameters.jl")
include("layers/conv2d.jl")
include("layers/pool.jl")
include("layers/matmul.jl")
include("layers/convlayer.jl")
include("layers/fullyconnectedlayer.jl")

include("models.jl")
include("utils/import_weights.jl")
include("utils/import_datasets.jl")
include("logging.jl")

function get_max_index(
    x::Array{T, 1})::Integer where {T<:Real}
    return findmax(x)[2]
end

(p::SoftmaxParameters)(x::Array{T, 1}) where {T<:JuMPReal} = p.mmparams(x)
(ps::Array{U, 1})(x::Array{T}) where {T<:JuMPReal, U<:Union{ConvolutionLayerParameters, FullyConnectedLayerParameters}} = (
    length(ps) == 0 ? x : ps[2:end](ps[1](x))
)
(p::StandardNeuralNetParameters)(x::Array{T, 4}) where {T<:JuMPReal} = (
    x |> p.convlayer_params |> MIPVerify.flatten |> p.fclayer_params |> p.softmax_params
)

"""
Permute dimensions of array because Python flattens arrays in the opposite order.
"""
function flatten(x::Array{T, N}) where {T, N}
    return permutedims(x, N:-1:1)[:]
end

function find_adversarial_example(
    nnparams::NeuralNetParameters, 
    input::Array{T, N},
    target_label::Int,
    solver_type::DataType;
    pp::PerturbationParameters = AdditivePerturbationParameters(),
    norm_order::Real = 1,
    tolerance = 0.0,
    rebuild::Bool = true)::Dict where {T<:Real, N}

    d = get_model(nnparams, input, pp, solver_type, rebuild)
    m = d[:Model]
    # Set perturbation objective
    @objective(m, Min, get_norm(norm_order, d[:Perturbation]))

    # Set output constraint
    set_max_index(d[:Output], target_label, tolerance)
    info(get_logger(current_module()), "Attempting to find adversarial example. Neural net predicted label is $(input |> nnparams |> get_max_index), target label is $target_label")
    d[:SolveStatus] = solve(m)
    return d
end

function get_label(y::Array{T, 1}, test_index::Int)::Int where {T<:Real}
    return y[test_index]
end

function get_input(x::Array{T, 4}, test_index::Int)::Array{T, 4} where {T<:Real}
    return x[test_index:test_index, :, :, :]
end

# Maybe merge functionality?


function get_norm(
    norm_order::Real,
    v::Array{T}) where {T<:Real}
    if norm_order == 1
        return sum(abs.(v))
    elseif norm_order == 2
        return sqrt(sum(v.*v))
    elseif norm_order == Inf
        return maximum(Iterators.flatten(abs.(v)))
    end
end

function get_norm(
    norm_order::Real,
    v::Array{T}) where {T<:JuMP.AbstractJuMPScalar}
    if norm_order == 1
        abs_v = abs_ge.(v)
        return sum(abs_v)
    elseif norm_order == 2
        return sum(v.*v)
    elseif norm_order == Inf
        return MIPVerify.maximum(abs_ge.(v) |> MIPVerify.flatten; tighten = false)
    end
end

include("../test/IntegrationTestHelpers.jl")

end

