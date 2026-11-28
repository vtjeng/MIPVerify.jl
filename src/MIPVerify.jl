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

export find_adversarial_example, frac_correct, interval_arithmetic, lp, mip, target_lp

@enum TighteningAlgorithm interval_arithmetic = 1 lp = 2 mip = 3 target_lp = 4

"""
Objective used when searching for an adversarial example.

`closest` minimizes perturbation distance, `worst` maximizes the target margin, and `feasibility`
asks only whether any input satisfies the target and perturbation constraints.
"""
@enum AdversarialExampleObjective closest = 1 worst = 2 feasibility = 3
const DEFAULT_TIGHTENING_ALGORITHM = target_lp
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
include("backward_bounds.jl")

function get_max_index(x::Array{<:Real,1})::Integer
    return findmax(x)[2]
end

"""
    get_target_margin(output, target_indexes)

Return the largest target logit minus the largest non-target logit. If every
index is targeted, return `Inf` because the target condition is vacuous.
Throw `ArgumentError` when `target_indexes` is empty or contains an index outside `output`.
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

"""
    witness_satisfies_target(output, target_indexes, required_margin)

Return the observed target margin and whether it meets `required_margin`. The check fails for any
non-finite output. Values within the witness verification tolerances of `required_margin` pass.
"""
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

const WITNESS_VALUE_KEYS =
    (:PerturbedInputValue, :WitnessOutput, :WitnessMargin, :WitnessDistance, :WitnessBlurKernel)

"""
    clear_witness_values!(result)

Remove every witness value recorded by an earlier evaluation. The witness flags always stay
present; witness values exist only while a candidate is available.
"""
function clear_witness_values!(d::Dict)::Nothing
    for key in WITNESS_VALUE_KEYS
        pop!(d, key, nothing)
    end
    return nothing
end

"""
    record_no_witness!(result)

Mutate `result` to record that no numeric witness is available. Remove any stale witness values
and set all witness verification flags to `false`.
"""
function record_no_witness!(d::Dict)::Nothing
    clear_witness_values!(d)
    d[:WitnessAvailable] = false
    d[:WitnessTargetVerified] = false
    d[:WitnessPerturbationVerified] = false
    d[:WitnessVerified] = false
    return nothing
end

"""
    record_witness!(result, nn, input, perturbed_input, pp, required_margin)

Independently evaluate and record a numeric witness candidate. The candidate is marked available
even when its target or perturbation check fails; `:WitnessVerified` is true only when both pass.
This function records the candidate, network output, observed margin, verification flags, and any
perturbation-specific values. Stale values from an earlier evaluation are removed first, so
verification never reads a previously recorded witness. The caller records `:WitnessDistance` when
applicable.
"""
function record_witness!(
    d::Dict,
    nn::NeuralNet,
    input::Array{<:Real},
    perturbed_input::Array{<:Real},
    pp::PerturbationFamily,
    margin::Real,
)::Nothing
    clear_witness_values!(d)
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

# A screen-only feasibility result has a mathematical termination status but no optimizer or
# JuMP model. Keep the original-input shortcut distinguishable by leaving its status absent.
function screen_feasibility!(d, nn, input, pp, norm_order, non_targets, margin)
    screen = BackwardBounds.screen(
        nn,
        input,
        non_targets,
        d[:TargetIndexes];
        radius = pp.norm_bound,
        margin,
    )
    isnothing(screen) && return nothing
    if maximum(screen.bounds) < margin
        d[:VerificationSource] = :backward_bound
        d[:SolveStatus] = MathOptInterface.INFEASIBLE
        d[:PrimalStatus] = MathOptInterface.NO_SOLUTION
        d[:ObjectiveValue] = d[:ObjectiveBound] = NaN
    else
        checked = BackwardBounds.search_checked_witness(screen, input)
        isnothing(checked) && return screen
        record_witness!(d, nn, input, checked.witness, pp, margin)
        if !d[:WitnessVerified]
            record_no_witness!(d)
            return screen
        end
        d[:VerificationSource] = checked.source == "corner" ? :corner : :pgd
        d[:SolveStatus] = MathOptInterface.OPTIMAL
        d[:PrimalStatus] = MathOptInterface.FEASIBLE_POINT
        d[:ObjectiveValue] = d[:ObjectiveBound] = 0.0
        d[:PerturbationValue] = checked.witness - input
        d[:WitnessDistance] = get_norm(norm_order, d[:PerturbationValue])
    end
    d[:PerturbationFamily] = pp
    d[:TighteningApproach] = "backward_bound"
    d[:SolveTime] = 0.0
    return screen
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
absolute tolerance so it does not grow with the channel count. These verification tolerances are
deliberately stricter than typical solver primal feasibility tolerances (HiGHS defaults to `1e-7`
and Gurobi to `1e-6`), so a solver can return an incumbent that sits on a constraint boundary yet
fails verification. Such a result is unresolved, not a solver wrong answer; tighten the solver's
feasibility tolerance if boundary-tight witnesses must verify. Solver-backed results also have keys
`:Model, :PerturbationFamily, :TargetIndexes, :SolveStatus, :PrimalStatus, :Perturbation,
:PerturbedInput, :Output`. `:AdversarialExampleObjective` records the selected search objective.
See the
[tutorial](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/03_interpreting_the_output_of_find_adversarial_example.ipynb)
on what individual dictionary entries correspond to.

*Formal Definition*: If there are a total of `n` categories, the numeric output vector
`y=d[:WitnessOutput]=d[:PerturbedInputValue] |> nn` has length `n`. If `:WitnessVerified` is true,
then `d[:PerturbedInputValue]` belongs to `pp` around `input`, and
`y[j] - y[i] ≥ margin` within the documented comparison tolerance for some
`j ∈ target_selection` and for all `i ∉ target_selection`. These claims are checked numerically;
a solver termination status alone does not establish them. Custom perturbation families fail this
check unless they implement `verify_perturbation_witness`.

# Keyword Arguments:
- `invert_target_selection`: Defaults to `false`. If `true`, sets `target_selection` to be its
    complement.
- `pp`: Defaults to `UnrestrictedPerturbationFamily()`. Determines the search space for adversarial
  examples.
- `norm_order`: Defaults to `1`. Determines the distance norm used to determine the distance from
    the perturbed image to the original. Allowed options are `1` and `Inf`, and `2` if the
    `optimizer` can solve MIQPs.
- `adversarial_example_objective`: Defaults to `closest`. Allowed options are `closest`, `worst`,
  or `feasibility`.
  - `closest` finds the closest adversarial example, as measured by the `norm_order` norm.
  - `worst` finds the adversarial example with the _largest_ gap between `max(y[j)` for `j ∈
    target_selection` and `max(y[i])` for all `i ∉ target_selection`.
  - `feasibility` asks whether any adversarial example satisfies the constraints. A returned
    `WitnessDistance` describes that candidate but is not necessarily minimal.
- `tightening_algorithm`: Defaults to `target_lp`. Selects how to bound nonlinear units and target
    margins. Allowed options are `interval_arithmetic`, `lp`, `mip`, and `target_lp`.
  - `interval_arithmetic` looks at the bounds on the output to the previous layer.
  - `lp` solves an `lp` corresponding to the `mip` formulation, but with any integer constraints
    relaxed.
  - `target_lp` uses interval bounds inside the network unless a layer specifies an override.
    For `feasibility` and `closest` with several targets and one non-target, it uses LP
    upper bounds on each target logit minus the non-target logit to eliminate impossible targets.
    Other target selections and the `worst` objective use interval bounds without this screening.
  - `mip` solves the full `mip` formulation.
- `backward_screen`: Defaults to `true`. For fixed-budget `feasibility`, first try backward bounds
    and checked candidate inputs, with any `tightening_algorithm`. The screen supports sequential
    Conv2d/Linear and ReLU networks, Float64 pixels, a finite Float64 L-infinity budget, and
    finite integer, rational or floating-point margins. It tests each target against the non-target
    classes. A checked witness meets the requested margin against every non-target on one input.
    When screening settles a case, `:VerificationSource` records `:backward_bound`, `:corner`,
    or `:pgd`, with a `:SolveStatus` and zero `:SolveTime`. Otherwise the JuMP model uses the
    selected tightening, per-layer overrides and MIP solver. Set `backward_screen=false` to skip
    the screen. Other objectives and unsupported networks use the solver formulation.
- `tightening_options`: Solver-specific options passed to optimizer when used to determine upper
    and lower bounds for input to nonlinear units. Note that these are only used if the
    `tightening_algorithm` is `lp`, `mip`, or `target_lp` (no solver is used when `interval_arithmetic` is used
    to compute the bounds). Defaults for Gurobi and HiGHS to a time limit of 20s per solve, with
    output suppressed.
- `solve_if_predicted_in_targeted`: Defaults to `true`. The prediction that `nn` makes for the
    unperturbed `input` can be determined efficiently. If the predicted index is one of the indexes
    in `target_selection`, we can skip the relatively costly process of building the model for the
    MIP problem since we already have an "adversarial example" --- namely, the input itself. We
    skip only when that input also passes the perturbation-family check. We continue building the
    model and solve the (trivial) MIP problem if `solve_if_predicted_in_targeted` is `true`, the
    requested margin is not met, or perturbation-family membership cannot be verified.
- `margin`: Defaults to `0.0`. If specified, the target category must have logits strictly larger
    (by at least `margin`) than any non-target category. 
- `collect_stats`: Defaults to `false`. If `true`, records formulation structure, progressive bound
    tightening work, ReLU stability, and main-solver work in the returned dictionary. The
    statistics keys are absent when a screen or original-input check finishes without building a model.
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
    backward_screen::Bool = true,
    margin::Real = 0.0,
    collect_stats::Bool = false,
)::Dict

    total_time = @elapsed begin
        d = Dict{Symbol,Any}(:AdversarialExampleObjective => adversarial_example_objective)
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
        screen = nothing
        if backward_screen &&
           !skip_solve &&
           (!solve_if_predicted_in_targeted || !(predicted_index in d[:TargetIndexes])) &&
           adversarial_example_objective == feasibility &&
           pp isa LInfNormBoundedPerturbationFamily &&
           pp.norm_bound isa Float64 &&
           isfinite(pp.norm_bound) &&
           margin isa Union{Integer,AbstractFloat,Rational} &&
           isfinite(margin)
            non_targets = setdiff(eachindex(predicted_output), d[:TargetIndexes])
            screen = screen_feasibility!(d, nn, input, pp, norm_order, non_targets, margin)
        end
        if skip_solve
            d[:WitnessDistance] = zero(float(norm_order))
        elseif !haskey(d, :SolveStatus)
            formulation_start = time_ns()
            model_network =
                isnothing(screen) ? nn :
                BackwardBounds.bounded_fallback_network(nn, screen.hidden, tightening_algorithm)
            merge!(
                d,
                get_model(
                    model_network,
                    input,
                    pp,
                    optimizer,
                    tightening_options,
                    tightening_algorithm,
                    collect_stats,
                ),
            )
            m = d[:Model]

            if adversarial_example_objective == feasibility
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
                checked =
                    isnothing(screen) ? nothing :
                    BackwardBounds.repair_witness(screen, input, perturbed_input)
                if !isnothing(checked)
                    perturbed_input = checked.witness
                    d[:PerturbationValue] = perturbed_input - input
                end
                record_witness!(d, nn, input, perturbed_input, pp, margin)
                if !isnothing(screen) && isnothing(checked)
                    # If the raw point is inside the checked box, repair tested its scores
                    # directly before trying other points. Otherwise its coordinates failed.
                    lo, hi = BackwardBounds.witness_box(
                        input;
                        radius = screen.radius,
                        decimal_radius = false,
                    )
                    if all(lo .<= vec(perturbed_input) .<= hi)
                        d[:WitnessTargetVerified] = false
                    else
                        d[:WitnessPerturbationVerified] = false
                    end
                    d[:WitnessVerified] = false
                end
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
