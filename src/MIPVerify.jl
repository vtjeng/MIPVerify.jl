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

function num_correct(nnparams::NeuralNetParameters, dataset_name::String, num_samples::Int)::Int
    """
    Returns the number of correctly classified items our neural net obtains
    of the first `num_samples` for the test set of the name dataset.
    """

    d = read_datasets(dataset_name)

    num_correct = 0
    for sample_index in 1:num_samples
        x0 = get_image(d.test.images, sample_index)
        actual_label = get_label(d.test.labels, sample_index)
        predicted_label = (x0 |> nnparams |> get_max_index) - 1
        if actual_label == predicted_label
            num_correct += 1
        end
    end
    return num_correct
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

