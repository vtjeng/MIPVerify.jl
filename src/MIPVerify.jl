module MIPVerify

using Base.Cartesian
using JuMP
using ConditionalJuMP
using Memento
using AutoHashEquals
using DocStringExtensions

const dependencies_path = joinpath(Pkg.dir("MIPVerify"), "deps")

include("net_components/main.jl")

include("models.jl")
include("utils/prep_data_file.jl")
include("utils/import_weights.jl")
include("utils/import_datasets.jl")
include("logging.jl")

export find_adversarial_example, frac_correct

function get_max_index(
    x::Array{<:Real, 1})::Integer
    return findmax(x)[2]
end

"""
Permute dimensions of array because Python flattens arrays in the opposite order.
"""
function flatten(x::Array{T, N}) where {T, N}
    return permutedims(x, N:-1:1)[:]
end

function get_default_model_build_solver(
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver
    )::MathProgBase.SolverInterface.AbstractMathProgSolver
    model_build_solver = typeof(main_solver)()
    MathProgBase.setparameters!(model_build_solver, Silent = true, TimeLimit = 20)
    return model_build_solver
end

"""
$(SIGNATURES)

Finds the perturbed image closest to `input` such that the network described by `nnparams`
classifies the perturbed image in one of the categories identified by the 
indexes in `target_selection`.

`main_solver` specifies the solver used.

*Formal Definition*: If there are a total of `n` categories, the output vector `y` has 
length `n`. We guarantee that `y[j] - y[i] ≥ tolerance` for some `j ∈ target_selection` 
and for all `i ∉ target_selection`.

# Named Arguments:
+ `pp::PerturbationParameters`: Defaults to `AdditivePerturbationParameters()`. Determines
    the family of perturbations over which we are searching for adversarial examples.
+ `norm_order::Real`: Defaults to `1`. Determines the distance norm used to determine the 
    distance from the perturbed image to the original. Supported options are `1`, `Inf` 
    and `2` (if the `main_solver` used can solve MIQPs.)
+ `tolerance`: Defaults to `0.0`. As above.
+ `rebuild`: Defaults to `false`. If `true`, rebuilds model by determining upper and lower
    bounds on input to each non-linear unit even if a cached model exists.
+ `invert_target_selection`: defaults to `false`. If `true`, sets `target_selection` to 
    be its complement.
+ `model_build_solver`: Used to determine the upper and lower bounds on input to each 
    non-linear unit. Defaults to the same type of solver as the `main_solver`, with a
    time limit of 20s per solver and output suppressed. 
"""
function find_adversarial_example(
    nnparams::NeuralNetParameters, 
    input::Array{<:Real},
    target_selection::Union{Integer, Array{<:Integer, 1}},
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver;
    pp::PerturbationParameters = AdditivePerturbationParameters(),
    norm_order::Real = 1,
    tolerance = 0.0,
    rebuild::Bool = false,
    invert_target_selection::Bool = false,
    model_build_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = get_default_model_build_solver(main_solver)
    )::Dict

    d = get_model(nnparams, input, pp, main_solver, model_build_solver, rebuild)
    m = d[:Model]

    # Set output constraint
    d[:TargetIndexes] = get_target_indexes(target_selection, length(d[:Output]), invert_target_selection = invert_target_selection)
    set_max_indexes(d[:Output], d[:TargetIndexes], tolerance=tolerance)

    notice(MIPVerify.LOGGER, "Attempting to find adversarial example. Neural net predicted label is $(input |> nnparams |> get_max_index), target labels are $(d[:TargetIndexes])")

    # Set perturbation objective
    # NOTE (vtjeng): It is important to set the objective immediately before we carry out
    # the solve. Functions like `set_max_indexes` can modify the objective.
    @objective(m, Min, get_norm(norm_order, d[:Perturbation]))
    d[:SolveStatus] = solve(m)
    return d
end

function get_label(y::Array{<:Real, 1}, test_index::Int)::Int
    return y[test_index]
end

function get_image(x::Array{T, 4}, test_index::Int)::Array{T, 4} where {T<:Real}
    return x[test_index:test_index, :, :, :]
end

"""
$(SIGNATURES)

Returns the fraction of items the neural network correctly classifies of the first
`num_samples` of the provided `dataset`. If there are fewer than
`num_samples` items, we use all of the available samples.

# Named Arguments:
+ `nnparams::NeuralNetParameters`: The parameters of the neural network.
+ `dataset::ImageDataset`:
+ `num_samples::Int`: Number of samples to use.
"""
function frac_correct(
    nnparams::NeuralNetParameters, 
    dataset::ImageDataset, 
    num_samples::Int)::Real

    num_correct = 0.0
    num_samples = min(num_samples, length(dataset.labels))
    for sample_index in 1:num_samples
        x0 = get_image(dataset.images, sample_index)
        actual_label = get_label(dataset.labels, sample_index)
        predicted_label = (x0 |> nnparams |> get_max_index) - 1
        if actual_label == predicted_label
            num_correct += 1
        end
    end
    return num_correct / num_samples
end


function get_norm(
    norm_order::Real,
    v::Array{<:Real})
    if norm_order == 1
        return sum(abs.(v))
    elseif norm_order == 2
        return sqrt(sum(v.*v))
    elseif norm_order == Inf
        return Base.maximum(Iterators.flatten(abs.(v)))
    else
        throw(DomainError("Only l1, l2 and l∞ norms supported."))
    end
end

function get_norm(
    norm_order::Real,
    v::Array{<:JuMP.AbstractJuMPScalar})
    if norm_order == 1
        abs_v = abs_ge.(v)
        return sum(abs_v)
    elseif norm_order == 2
        return sum(v.*v)
    elseif norm_order == Inf
        return MIPVerify.maximum(abs_ge.(v) |> MIPVerify.flatten; tighten = false)
    else
        throw(DomainError("Only l1, l2 and l∞ norms supported."))
    end
end

end

