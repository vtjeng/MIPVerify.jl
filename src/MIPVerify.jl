module MIPVerify

using Base.Cartesian
using JuMP
using MathOptInterface
using Memento
using DocStringExtensions
using ProgressMeter

# TODO: more reliable way to determine location for dependencies
const dependencies_path = joinpath(@__DIR__, "..", "deps")

export find_adversarial_example, frac_correct, interval_arithmetic, lp, mip

@enum TighteningAlgorithm interval_arithmetic = 1 lp = 2 mip = 3
@enum AdversarialExampleObjective closest = 1 worst = 2
const DEFAULT_TIGHTENING_ALGORITHM = mip

# importing vendor/ConditionalJuMP.jl first as the remaining files use functions
# defined in it. we're unsure if this is necessary.
include("vendor/ConditionalJuMP.jl")
include("net_components.jl")
include("models.jl")
include("utils.jl")
include("logging.jl")

function get_max_index(x::Array{<:Real,1})::Integer
    return findmax(x)[2]
end

function get_default_tightening_options(optimizer)::Dict
    optimizer_type_name = string(typeof(optimizer()))
    if optimizer_type_name == "Gurobi.Optimizer"
        return Dict("OutputFlag" => 0, "TimeLimit" => 20)
    elseif optimizer_type_name == "Cbc.Optimizer"
        return Dict("logLevel" => 0, "seconds" => 20)
    else
        return Dict()
    end
end

"""
$(SIGNATURES)

Finds the perturbed image closest to `input` such that the network described by `nn`
classifies the perturbed image in one of the categories identified by the
indexes in `target_selection`.

`optimizer` specifies the optimizer used to solve the MIP problem once it has been built.

The output dictionary has keys `:Model, :PerturbationFamily, :TargetIndexes, :SolveStatus,
:Perturbation, :PerturbedInput, :Output`.
See the [tutorial](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/03_interpreting_the_output_of_find_adversarial_example.ipynb)
on what individual dictionary entries correspond to.

*Formal Definition*: If there are a total of `n` categories, the (perturbed) output vector
`y=d[:Output]=d[:PerturbedInput] |> nn` has length `n`.
We guarantee that `y[j] - y[i] ≥ 0` for some `j ∈ target_selection` and for all 
`i ∉ target_selection`.

# Named Arguments:
+ `invert_target_selection::Bool`: Defaults to `false`. If `true`, sets `target_selection` to
    be its complement.
+ `pp::PerturbationFamily`: Defaults to `UnrestrictedPerturbationFamily()`. Determines
    the family of perturbations over which we are searching for adversarial examples.
+ `norm_order::Real`: Defaults to `1`. Determines the distance norm used to determine the
    distance from the perturbed image to the original. Supported options are `1`, `Inf`
    and `2` (if the `optimizer` used can solve MIQPs.)
+ `tightening_algorithm::MIPVerify.TighteningAlgorithm`: Defaults to `mip`. Determines how we
    determine the upper and lower bounds on input to each nonlinear unit.
    Allowed options are `interval_arithmetic`, `lp`, `mip`.
    (1) `interval_arithmetic` looks at the bounds on the output to the previous layer.
    (2) `lp` solves an `lp` corresponding to the `mip` formulation, but with any integer constraints
         relaxed.
    (3) `mip` solves the full `mip` formulation.
+ `tightening_options`: Solver-specific options passed to optimizer when used to determine upper and
    lower bounds for input to nonlinear units. Note that these are only used if the 
    `tightening_algorithm` is `lp` or `mip` (no solver is used when `interval_arithmetic` is used
    to compute the bounds). Defaults for Gurobi and Cbc to a time limit of 20s per solve, 
    with output suppressed.
+ `solve_if_predicted_in_targeted`: Defaults to `true`. The prediction that `nn` makes for the 
    unperturbed `input` can be determined efficiently. If the predicted index is one of the indexes 
    in `target_selection`, we can skip the relatively costly process of building the model for the 
    MIP problem since we already have an "adversarial example" --- namely, the input itself. We 
    continue build the model and solve the (trivial) MIP problem if and only if 
    `solve_if_predicted_in_targeted` is `true`.
"""
function find_adversarial_example(
    nn::NeuralNet,
    input::Array{<:Real},
    target_selection::Union{Integer,Array{<:Integer,1}},
    optimizer,
    main_solve_options::Dict;
    invert_target_selection::Bool = false,
    pp::PerturbationFamily = UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    adversarial_example_objective::AdversarialExampleObjective = closest,
    tightening_algorithm::TighteningAlgorithm = DEFAULT_TIGHTENING_ALGORITHM,
    tightening_options::Dict = get_default_tightening_options(optimizer),
    solve_if_predicted_in_targeted = true,
    silence_solve_output::Bool = false,
)::Dict

    total_time = @elapsed begin
        d = Dict()

        # Calculate predicted index
        predicted_output = input |> nn
        num_possible_indexes = length(predicted_output)
        predicted_index = predicted_output |> get_max_index

        d[:PredictedIndex] = predicted_index

        # Set target indexes
        d[:TargetIndexes] = get_target_indexes(
            target_selection,
            num_possible_indexes,
            invert_target_selection = invert_target_selection,
        )
        notice(
            MIPVerify.LOGGER,
            "Attempting to find adversarial example. Neural net predicted label is $(predicted_index), target labels are $(d[:TargetIndexes])",
        )

        # Only call optimizer if predicted index is not found among target indexes.
        if !(d[:PredictedIndex] in d[:TargetIndexes]) || solve_if_predicted_in_targeted
            merge!(d, get_model(nn, input, pp, optimizer, tightening_options, tightening_algorithm))
            m = d[:Model]

            if adversarial_example_objective == closest
                set_max_indexes(m, d[:Output], d[:TargetIndexes])

                # Set perturbation objective
                # NOTE (vtjeng): It is important to set the objective immediately before we carry
                # out the solve. Functions like `set_max_indexes` can modify the objective.
                @objective(m, Min, get_norm(norm_order, d[:Perturbation]))
            elseif adversarial_example_objective == worst
                (maximum_target_var, nontarget_vars) =
                    get_vars_for_max_index(d[:Output], d[:TargetIndexes])
                maximum_nontarget_var = maximum_ge(nontarget_vars)
                # Introduce an additional variable since Gurobi ignores constant terms in objective, 
                # but we explicitly need these if we want to stop early based on the value of the 
                # objective (not simply whether or not it is maximized).
                # See discussion in https://github.com/jump-dev/Gurobi.jl/issues/111 for more 
                # details.
                v_obj = @variable(m)
                @constraint(m, v_obj == maximum_target_var - maximum_nontarget_var)
                @objective(m, Max, v_obj)
            else
                error("Unknown adversarial_example_objective $adversarial_example_objective")
            end
            set_optimizer(m, optimizer)
            set_optimizer_attributes(m, main_solve_options...)
            if silence_solve_output
                optimize_silent!(m)
            else
                optimize!(m)
            end
            d[:SolveStatus] = JuMP.termination_status(m)
            d[:SolveTime] = JuMP.solve_time(m)
        end
    end

    d[:TotalTime] = total_time
    return d
end

function get_label(y::Array{<:Real,1}, test_index::Integer)::Int
    return y[test_index]
end

function get_image(x::Array{T,4}, test_index::Integer)::Array{T,4} where {T<:Real}
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
+ `num_samples::Integer`: Number of samples to use.
"""
function frac_correct(nn::NeuralNet, dataset::LabelledDataset, num_samples::Integer)::Real

    num_correct = 0.0
    num_samples = min(num_samples, MIPVerify.num_samples(dataset))
    p = Progress(num_samples, desc = "Computing fraction correct...", enabled = isinteractive())
    for sample_index in 1:num_samples
        input = get_image(dataset.images, sample_index)
        actual_label = get_label(dataset.labels, sample_index)
        predicted_label = (input |> nn |> get_max_index) - 1
        if actual_label == predicted_label
            num_correct += 1
        end
        next!(p)
    end
    return num_correct / num_samples
end


function get_norm(norm_order::Real, v::Array{<:Real})
    if norm_order == 1
        return sum(abs.(v))
    elseif norm_order == 2
        return sqrt(sum(v .* v))
    elseif norm_order == Inf
        return Base.maximum(Iterators.flatten(abs.(v)))
    else
        throw(DomainError("Only l1, l2 and l∞ norms supported."))
    end
end

function get_norm(norm_order::Real, v::Array{<:JuMPLinearType,N}) where {N}
    if norm_order == 1
        abs_v = abs_ge.(v)
        return sum(abs_v)
    elseif norm_order == 2
        return sum(v .* v)
    elseif norm_order == Inf
        return MIPVerify.maximum_ge(permute_and_flatten(abs_ge.(v), N:-1:1))
    else
        throw(DomainError("Only l1, l2 and l∞ norms supported."))
    end
end

include("batch_processing_helpers.jl")

end
