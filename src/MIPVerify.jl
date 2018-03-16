module MIPVerify

using Base.Cartesian
using JuMP
using ConditionalJuMP
using Memento
using AutoHashEquals
using DocStringExtensions
using ProgressMeter
using CSV
using DataFrames

const dependencies_path = joinpath(Pkg.dir("MIPVerify"), "deps")

export find_adversarial_example, frac_correct, interval_arithmetic, lp, mip
@enum TighteningAlgorithm interval_arithmetic=1 lp=2 mip=3
@enum BatchRerunOption skip_existing=1 redo_existing=2 give_more_time=3

include("net_components.jl")
include("models.jl")
include("utils.jl")
include("logging.jl")

function get_max_index(
    x::Array{<:Real, 1})::Integer
    return findmax(x)[2]
end

function get_default_tightening_solver(
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver
    )::MathProgBase.SolverInterface.AbstractMathProgSolver
    tightening_solver = typeof(main_solver)()
    MathProgBase.setparameters!(tightening_solver, Silent = true, TimeLimit = 20)
    return tightening_solver
end

"""
$(SIGNATURES)

Finds the perturbed image closest to `input` such that the network described by `nn`
classifies the perturbed image in one of the categories identified by the 
indexes in `target_selection`.

`main_solver` specifies the solver used.

*Formal Definition*: If there are a total of `n` categories, the output vector `y` has 
length `n`. We guarantee that `y[j] - y[i] ≥ tolerance` for some `j ∈ target_selection` 
and for all `i ∉ target_selection`.

# Named Arguments:
+ `pp::PerturbationFamily`: Defaults to `UnrestrictedPerturbationFamily()`. Determines
    the family of perturbations over which we are searching for adversarial examples.
+ `norm_order::Real`: Defaults to `1`. Determines the distance norm used to determine the 
    distance from the perturbed image to the original. Supported options are `1`, `Inf` 
    and `2` (if the `main_solver` used can solve MIQPs.)
+ `tolerance`: Defaults to `0.0`. As above.
+ `rebuild`: Defaults to `false`. If `true`, rebuilds model by determining upper and lower
    bounds on input to each non-linear unit even if a cached model exists.
+ `invert_target_selection`: Defaults to `false`. If `true`, sets `target_selection` to 
    be its complement.
+ `tightening_algorithm`: Defaults to `lp`. Determines how we determine the upper and lower
    bounds on input to each nonlinear unit. Allowed options are `interval_arithmetic`, `lp`, `mip`.
   (1) `interval_arithmetic` looks at the bounds on the output to the previous layer.
   (2) `lp` solves an `lp` corresponding to the `mip` formulation, but with any integer constraints relaxed.
   (3) `mip` solves the full `mip` formulation.
+ `tightening_solver`: Defaults to the same type of solver as
    the `main_solver`, with a time limit of 20s per solver and output suppressed. Used only
    if the `tightening_algorithm` is `lp` or `mip`.
"""
function find_adversarial_example(
    nn::NeuralNet, 
    input::Array{<:Real},
    target_selection::Union{Integer, Array{<:Integer, 1}},
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver;
    pp::PerturbationFamily = UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    tolerance::Real = 0.0,
    rebuild::Bool = false,
    invert_target_selection::Bool = false,
    tightening_algorithm::TighteningAlgorithm = lp,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = get_default_tightening_solver(main_solver)
    )::Dict

    d = get_model(nn, input, pp, main_solver, tightening_solver, rebuild, tightening_algorithm)
    m = d[:Model]

    # Set output constraint
    d[:TargetIndexes] = get_target_indexes(target_selection, length(d[:Output]), invert_target_selection = invert_target_selection)
    set_max_indexes(d[:Output], d[:TargetIndexes], tolerance=tolerance)

    notice(MIPVerify.LOGGER, "Attempting to find adversarial example. Neural net predicted label is $(input |> nn |> get_max_index), target labels are $(d[:TargetIndexes])")

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
+ `nn::NeuralNet`: The parameters of the neural network.
+ `dataset::LabelledDataset`:
+ `num_samples::Int`: Number of samples to use.
"""
function frac_correct(
    nn::NeuralNet, 
    dataset::LabelledDataset, 
    num_samples::Int)::Real

    num_correct = 0.0
    num_samples = min(num_samples, MIPVerify.num_samples(dataset))
    @showprogress 1 "Computing fraction correct..." for sample_index in 1:num_samples
        x0 = get_image(dataset.images, sample_index)
        actual_label = get_label(dataset.labels, sample_index)
        predicted_label = (x0 |> nn |> get_max_index) - 1
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
    v::Array{<:JuMPLinearType, N}) where {N}
    if norm_order == 1
        abs_v = abs_ge.(v)
        return sum(abs_v)
    elseif norm_order == 2
        return sum(v.*v)
    elseif norm_order == Inf
        return MIPVerify.maximum_ge(flatten(abs_ge.(v), N:-1:1))
    else
        throw(DomainError("Only l1, l2 and l∞ norms supported."))
    end
end

include("batch_processing_helpers.jl")

end

