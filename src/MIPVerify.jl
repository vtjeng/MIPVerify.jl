module MIPVerify

using Base.Cartesian
using JuMP
using MathOptInterface
using Memento
using DocStringExtensions
using ProgressMeter

function resolve_dependencies_path(;
    env::AbstractDict = ENV,
    package_root::Union{String,Nothing} = pkgdir(@__MODULE__),
)::String
    env_override = get(env, "MIPVERIFY_DEPS_PATH", "")
    if !isempty(strip(env_override))
        return abspath(env_override)
    end
    if package_root === nothing
        return normpath(joinpath(@__DIR__, "..", "deps"))
    end
    return joinpath(package_root, "deps")
end

const dependencies_path = resolve_dependencies_path()

export find_adversarial_example, frac_correct, interval_arithmetic, lp, mip

@enum TighteningAlgorithm interval_arithmetic = 1 lp = 2 mip = 3
@enum AdversarialExampleObjective closest = 1 worst = 2
const DEFAULT_TIGHTENING_ALGORITHM = mip
const WITNESS_VERIFICATION_ATOL = 1e-8
const WITNESS_VERIFICATION_RTOL = 1e-8

include("instrumentation.jl")

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

"""
    get_target_margin(output, target_indexes)

Return the largest target logit minus the largest non-target logit. If every
index is targeted, return `Inf` because the target condition is vacuous.
"""
function get_target_margin(
    output::AbstractVector{<:Real},
    target_indexes::AbstractVector{<:Integer},
)::Real
    isempty(target_indexes) && throw(ArgumentError("target_indexes must not be empty"))
    all(i -> i in eachindex(output), target_indexes) ||
        throw(ArgumentError("target_indexes contains an index outside output"))

    is_target = [i in target_indexes for i in eachindex(output)]
    maximum_target = maximum(output[is_target])
    all(is_target) && return Inf
    return maximum_target - maximum(output[.!is_target])
end

function witness_satisfies_target(
    output::AbstractVector{<:Real},
    target_indexes::AbstractVector{<:Integer},
    margin::Real,
)::Tuple{Real,Bool}
    witness_margin = get_target_margin(output, target_indexes)
    finite_output = all(isfinite, output)
    verified =
        finite_output && (
            witness_margin >= margin || isapprox(
                witness_margin,
                margin;
                atol = WITNESS_VERIFICATION_ATOL,
                rtol = WITNESS_VERIFICATION_RTOL,
            )
        )
    return witness_margin, verified
end

function record_no_witness!(d::Dict)::Nothing
    for key in
        (:PerturbedInputValue, :WitnessOutput, :WitnessMargin, :WitnessDistance, :WitnessBlurKernel)
        pop!(d, key, nothing)
    end
    d[:WitnessAvailable] = false
    d[:WitnessTargetVerified] = false
    d[:WitnessPerturbationVerified] = false
    d[:WitnessVerified] = false
    d[:WitnessMargin] = missing
    return nothing
end

function record_witness!(
    d::Dict,
    nn::NeuralNet,
    input::Array{<:Real},
    perturbed_input::Array{<:Real},
    pp::PerturbationFamily,
    margin::Real,
)::Nothing
    witness_output = perturbed_input |> nn
    witness_margin, satisfies_target =
        witness_satisfies_target(witness_output, d[:TargetIndexes], margin)
    satisfies_perturbation, perturbation_values =
        verify_perturbation_witness(pp, input, perturbed_input, d)
    merge!(d, perturbation_values)
    finite_input =
        size(input) == size(perturbed_input) &&
        all(isfinite, input) &&
        all(isfinite, perturbed_input)
    d[:WitnessAvailable] = true
    d[:WitnessTargetVerified] = satisfies_target
    d[:WitnessPerturbationVerified] = finite_input && satisfies_perturbation
    d[:WitnessVerified] = d[:WitnessTargetVerified] && d[:WitnessPerturbationVerified]
    d[:PerturbedInputValue] = perturbed_input
    d[:WitnessOutput] = witness_output
    d[:WitnessMargin] = witness_margin
    return nothing
end

function get_default_tightening_options(optimizer)::Dict
    optimizer_type_name = string(typeof(optimizer()))
    if optimizer_type_name == "Gurobi.Optimizer"
        return Dict("OutputFlag" => 0, "TimeLimit" => 20)
    elseif optimizer_type_name == "HiGHS.Optimizer"
        return Dict("output_flag" => false, "time_limit" => 20.0)
    else
        return Dict()
    end
end

"""
$(SIGNATURES)

Perturbs `input` such that the network `nn` classifies the perturbed image in one of the categories
identified by the indexes in `target_selection`.

IMPORTANT: 
  1) `target_selection` can include the correct label for `input`.
  2) It is possible (particularly with the `closest` objective) to see 'ties' -- that is, the
     perturbed input produces an output with two logits (one corresponding to a target category,
     and one corresponding to a non-target category) taking on the same maximal value. See the
     formal definition below for more; in particular, note that '≥' sign.

`optimizer` is used to build and solve the MIP problem.

The output dictionary records whether a numeric incumbent was available in `:WitnessAvailable`.
`:WitnessTargetVerified` records whether an independent network forward pass satisfied the target
condition. `:WitnessPerturbationVerified` records whether the numeric input satisfied the selected
perturbation family's input constraints. `:WitnessVerified` is true only when both checks pass for a
numeric candidate. Available witnesses also have numeric `:PerturbedInputValue`,
`:WitnessOutput`, `:WitnessMargin`, and `:WitnessDistance` entries. Numeric comparisons allow an
absolute or relative tolerance of `1e-8`. An L-infinity budget uses only the relative tolerance so
a small budget is not expanded by a larger fixed tolerance; a blur-kernel sum uses only the
absolute tolerance so it does not grow with the channel count. Solver-backed results also have keys
`:Model, :PerturbationFamily, :TargetIndexes, :SolveStatus, :PrimalStatus, :Perturbation,
:PerturbedInput, :Output`. See the
[tutorial](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/03_interpreting_the_output_of_find_adversarial_example.ipynb)
on what individual dictionary entries correspond to.

*Formal Definition*: If there are a total of `n` categories, the numeric output vector
`y=d[:WitnessOutput]=d[:PerturbedInputValue] |> nn` has length `n`. If `:WitnessVerified` is true,
then `d[:PerturbedInputValue]` belongs to `pp` around `input`, and
`y[j] - y[i] ≥ margin` within the documented comparison tolerance for some
`j ∈ target_selection` and for all `i ∉ target_selection`. These claims are checked numerically;
a solver termination status alone does not establish them. Custom perturbation families fail this
check unless they implement [`verify_perturbation_witness`](@ref).

# Keyword Arguments:
- `invert_target_selection`: Defaults to `false`. If `true`, sets `target_selection` to be its
    complement.
- `pp`: Defaults to `UnrestrictedPerturbationFamily()`. Determines the search space for adversarial
  examples.
- `norm_order`: Defaults to `1`. Determines the distance norm used to determine the distance from
    the perturbed image to the original. Allowed options are `1` and `Inf`, and `2` if the
    `optimizer` can solve MIQPs.
- `adversarial_example_objective`: Defaults to `closest`. Allowed options are `closest` or `worst`.
  - `closest` finds the closest adversarial example, as measured by the `norm_order` norm.
  - `worst` finds the adversarial example with the _largest_ gap between `max(y[j)` for `j ∈
    target_selection` and `max(y[i])` for all `i ∉ target_selection`.
- `tightening_algorithm`: Defaults to `mip`. Determines how we determine the upper and lower bounds
    on input to each nonlinear unit. Allowed options are `interval_arithmetic`, `lp`, `mip`.
  - `interval_arithmetic` looks at the bounds on the output to the previous layer.
  - `lp` solves an `lp` corresponding to the `mip` formulation, but with any integer constraints
    relaxed.
  - `mip` solves the full `mip` formulation.
- `tightening_options`: Solver-specific options passed to optimizer when used to determine upper
    and lower bounds for input to nonlinear units. Note that these are only used if the
    `tightening_algorithm` is `lp` or `mip` (no solver is used when `interval_arithmetic` is used
    to compute the bounds). Defaults for Gurobi and HiGHS to a time limit of 20s per solve, with
    output suppressed.
- `solve_if_predicted_in_targeted`: Defaults to `true`. The prediction that `nn` makes for the
    unperturbed `input` can be determined efficiently. If the predicted index is one of the indexes
    in `target_selection`, we can skip the relatively costly process of building the model for the
    MIP problem since we already have an "adversarial example" --- namely, the input itself. We
    skip only when that input also passes the perturbation-family check. We continue building the
    model and solve the (trivial) MIP problem if `solve_if_predicted_in_targeted` is `true`, the
    requested margin is not met, or perturbation-family membership cannot be verified.
- `verdict_only`: Defaults to `false`, which computes the exact objective optimum. If `true`, stops
    once the solver finds a feasible adversarial example. In either mode, a returned incumbent is
    independently checked against the perturbation constraints and with a numeric network forward
    pass before `:WitnessVerified` is true.
- `margin`: Defaults to `0.0`. If specified, the target category must have logits strictly larger
    (by at least `margin`) than any non-target category. 
- `collect_stats`: Defaults to `false`. If `true`, records formulation structure, progressive bound
    tightening work, ReLU stability, and main-solver work in the returned dictionary. The
    statistics keys are absent when no model is built (that is, when the predicted index is in
    `target_selection` and `solve_if_predicted_in_targeted` is `false`).
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
    verdict_only::Bool = false,
    margin::Real = 0.0,
    collect_stats::Bool = false,
)::Dict

    total_time = @elapsed begin
        d = Dict{Symbol,Any}(:VerdictOnly => verdict_only)
        record_no_witness!(d)

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

        _, original_input_is_witness =
            witness_satisfies_target(predicted_output, d[:TargetIndexes], margin)

        # Skip the optimizer only when the unperturbed input satisfies the complete target
        # condition, including a requested positive margin.
        skip_candidate =
            d[:PredictedIndex] in d[:TargetIndexes] &&
            !solve_if_predicted_in_targeted &&
            original_input_is_witness
        skip_solve = false
        if skip_candidate
            record_witness!(d, nn, input, copy(input), pp, margin)
            skip_solve = d[:WitnessVerified]
            skip_solve || record_no_witness!(d)
        end
        if skip_solve
            d[:WitnessDistance] = zero(float(norm_order))
        else
            formulation_start = time_ns()
            merge!(
                d,
                get_model(
                    nn,
                    input,
                    pp,
                    optimizer,
                    tightening_options,
                    tightening_algorithm,
                    collect_stats,
                ),
            )
            m = d[:Model]

            if verdict_only
                # A feasibility objective is solver-independent. Once a feasible point is found,
                # there is no remaining optimality gap to close.
                set_max_indexes(m, d[:Output], d[:TargetIndexes], margin = margin)
                JuMP.set_objective_sense(m, MathOptInterface.FEASIBILITY_SENSE)
            elseif adversarial_example_objective == closest
                set_max_indexes(m, d[:Output], d[:TargetIndexes], margin = margin)

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
                # JuMP does not support strict inequalities; see 
                # https://github.com/jump-dev/JuMP.jl/blob/24c0409c5fa5cae6a4ae64b1c82ab5f83d55fbc6/src/macros/%40variable.jl#L516-L523
                # for more context.
                @constraint(m, v_obj >= margin)
                @objective(m, Max, v_obj)
            else
                error("Unknown adversarial_example_objective $adversarial_example_objective")
            end
            set_optimizer(m, optimizer)
            set_optimizer_attributes(m, main_solve_options...)
            if collect_stats
                d[:FormulationTime] = elapsed_seconds(formulation_start)
                merge!(d, summarize_verification_stats(m))
                d[:FormulationExcludingBoundSolveTime] =
                    max(0.0, d[:FormulationTime] - d[:BoundSolverWallTime])
                d[:NumVariables] = JuMP.num_variables(m)
                d[:NumBinaryVariables] =
                    JuMP.num_constraints(m, JuMP.VariableRef, MathOptInterface.ZeroOne)
                d[:NumStructuralConstraints] =
                    JuMP.num_constraints(m; count_variable_in_set_constraints = false)
                d[:NumTotalConstraints] =
                    JuMP.num_constraints(m; count_variable_in_set_constraints = true)
            end
            main_solve_start = time_ns()
            optimize!(m)
            if collect_stats
                d[:MainSolveWallTime] = elapsed_seconds(main_solve_start)
            end
            d[:SolveStatus] = JuMP.termination_status(m)
            d[:PrimalStatus] = JuMP.primal_status(m)
            d[:SolveTime] = JuMP.solve_time(m)
            if d[:PrimalStatus] == MathOptInterface.FEASIBLE_POINT
                perturbed_input = JuMP.value.(d[:PerturbedInput])
                record_witness!(d, nn, input, perturbed_input, pp, margin)
                d[:WitnessDistance] = get_norm(norm_order, perturbed_input - input)
                if !d[:WitnessVerified]
                    Memento.warn(
                        MIPVerify.LOGGER,
                        "Solver returned a primal point, but the numeric witness did not verify. " *
                        "primal_status=$(d[:PrimalStatus]), witness_margin=$(d[:WitnessMargin]), " *
                        "required_margin=$margin, " *
                        "target_verified=$(d[:WitnessTargetVerified]), " *
                        "perturbation_verified=$(d[:WitnessPerturbationVerified])",
                    )
                end
            else
                record_no_witness!(d)
            end
            if collect_stats
                d[:MainNodeCount] = safe_solve_metric(JuMP.node_count, m, missing)
                d[:MainSimplexIterations] =
                    safe_solve_metric(JuMP.simplex_iterations, m, missing)
                d[:MainBarrierIterations] =
                    safe_solve_metric(JuMP.barrier_iterations, m, missing)
                d[:MainRelativeGap] = safe_solve_metric(JuMP.relative_gap, m, missing)
            end
        end
    end

    d[:TotalTime] = total_time
    return d
end

function get_label(y::Array{<:Real,1}, test_index::Integer)::Int
    return y[test_index]
end

function get_label(
    dataset::LabelledImageDataset{T,U},
    test_index::Integer,
)::Int where {T<:Real,U<:Integer}
    return get_label(dataset.labels, test_index)
end

function get_image(x::Array{T,4}, test_index::Integer)::Array{T,4} where {T<:Real}
    return x[test_index:test_index, :, :, :]
end

function get_image(
    dataset::LabelledImageDataset{T,U},
    test_index::Integer,
)::Array{T,4} where {T<:Real,U<:Integer}
    return get_image(dataset.images, test_index)
end

"""
$(SIGNATURES)

Returns the fraction of items the neural network correctly classifies of the first
`num_samples` of the provided `dataset`. If there are fewer than
`num_samples` items, we use all of the available samples.

# Named Arguments:
- `nn::NeuralNet`: The parameters of the neural network.
- `dataset::LabelledDataset`:
- `num_samples::Integer`: Number of samples to use.
"""
function frac_correct(nn::NeuralNet, dataset::LabelledDataset, num_samples::Integer)::Real

    num_correct = 0.0
    num_samples = min(num_samples, MIPVerify.num_samples(dataset))
    p = Progress(num_samples, desc = "Computing fraction correct...", enabled = isinteractive())
    for sample_index in 1:num_samples
        input = get_image(dataset, sample_index)
        actual_label = get_label(dataset, sample_index)
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
