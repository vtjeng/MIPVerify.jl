export batch_find_untargeted_attack

using DelimitedFiles, Dates, DataFrames, CSV, MathOptInterface

@enum SolveRerunOption never = 1 always = 2 resolve_ambiguous_cases = 3 refine_insecure_cases = 4 retarget_infeasible_cases =
    5

struct BatchRunParameters
    nn::NeuralNet
    pp::PerturbationFamily
    norm_order::Real
end

# BatchRunParameters are used for result folder names
function Base.show(io::IO, t::BatchRunParameters)
    print(io, "$(t.nn.UUID)__$(t.pp)__$(t.norm_order)")
end

function mkpath_if_not_present(path::String)
    !isdir(path) ? mkpath(path) : nothing
end

function extract_results_for_save(d::Dict)::Dict
    m = d[:Model]
    status = d[:SolveStatus]
    r = Dict()
    r[:SolveTime] = d[:SolveTime]
    r[:VerdictOnly] = d[:VerdictOnly]
    r[:WitnessAvailable] = d[:WitnessAvailable]
    r[:WitnessVerified] = d[:WitnessVerified]
    if d[:WitnessAvailable]
        r[:WitnessMargin] = d[:WitnessMargin]
        r[:WitnessOutput] = d[:WitnessOutput]
        r[:PerturbedInputValue] = d[:PerturbedInputValue]
        r[:WitnessDistance] = d[:WitnessDistance]
    end
    if !is_infeasible(status) && d[:WitnessAvailable]
        r[:ObjectiveBound] = try
            JuMP.objective_bound(m)
        catch
            NaN
        end
        r[:ObjectiveValue] = try
            JuMP.objective_value(m)
        catch
            NaN
        end
        r[:PerturbationValue] = d[:Perturbation] .|> JuMP.value
    elseif !is_infeasible(status)
        r[:ObjectiveBound] = try
            JuMP.objective_bound(m)
        catch
            NaN
        end
        r[:ObjectiveValue] = NaN
    else
        r[:ObjectiveBound] = NaN
        r[:ObjectiveValue] = NaN
    end
    r[:TargetIndexes] = d[:TargetIndexes]
    r[:SolveStatus] = d[:SolveStatus]
    r[:PredictedIndex] = d[:PredictedIndex]
    r[:TighteningApproach] = d[:TighteningApproach]
    r[:TotalTime] = d[:TotalTime]
    return r
end

const SUMMARY_HEADER = [
    "SampleNumber",
    "ResultRelativePath",
    "PredictedIndex",
    "TargetIndexes",
    "SolveTime",
    "SolveStatus",
    "IsInfeasible",
    "ObjectiveValue",
    "ObjectiveBound",
    "TighteningApproach",
    "TotalTime",
    "VerdictOnly",
    "WitnessAvailable",
    "WitnessVerified",
    "WitnessMargin",
]

function summary_witness_margin(d::Dict)
    return d[:WitnessAvailable] ? d[:WitnessMargin] : NaN
end

function is_infeasible(s::MathOptInterface.TerminationStatusCode)::Bool
    return s == MathOptInterface.INFEASIBLE
end

function get_uuid()::String
    Dates.format(now(), "yyyy-mm-ddTHH.MM.SS.sss")
end

function generate_csv_summary_line(
    sample_number::Integer,
    results_file_relative_path::String,
    r::Dict,
)
    [
        sample_number,
        results_file_relative_path,
        r[:PredictedIndex],
        r[:TargetIndexes],
        r[:SolveTime],
        r[:SolveStatus],
        r[:SolveStatus] |> is_infeasible,
        r[:ObjectiveValue],
        r[:ObjectiveBound],
        r[:TighteningApproach],
        r[:TotalTime],
        r[:VerdictOnly],
        r[:WitnessAvailable],
        r[:WitnessVerified],
        summary_witness_margin(r),
    ] .|> string
end

function generate_csv_summary_line_optimal(sample_number::Integer, d::Dict)
    @assert(d[:PredictedIndex] in d[:TargetIndexes])
    [
        sample_number,
        "",
        d[:PredictedIndex],
        d[:TargetIndexes],
        0,
        MathOptInterface.OPTIMAL,
        false,
        0,
        0,
        :NA,
        d[:TotalTime],
        d[:VerdictOnly],
        d[:WitnessAvailable],
        d[:WitnessVerified],
        summary_witness_margin(d),
    ] .|> string
end

function create_summary_file_if_not_present(summary_file_path::String)
    if !isfile(summary_file_path)
        open(summary_file_path, "w") do file
            writedlm(file, [SUMMARY_HEADER], ',')
        end
    end
end

function read_summary_file(summary_file_path::String)::DataFrames.DataFrame
    dt = DataFrame(CSV.File(summary_file_path))
    schema_changed = false
    if :IsInfeasible in propertynames(dt)
        proven_infeasible = string.(dt[!, :SolveStatus]) .== "INFEASIBLE"
        if dt[!, :IsInfeasible] != proven_infeasible
            dt[!, :IsInfeasible] = proven_infeasible
            schema_changed = true
        end
    end
    if :VerdictOnly ∉ propertynames(dt)
        dt[!, :VerdictOnly] = falses(nrow(dt))
        schema_changed = true
    end
    for column in (:WitnessAvailable, :WitnessVerified, :WitnessMargin)
        if column ∉ propertynames(dt)
            dt[!, column] = fill(missing, nrow(dt))
            schema_changed = true
        end
    end
    schema_changed && CSV.write(summary_file_path, dt)
    return dt
end

function verify_target_indices(
    target_indices::AbstractArray{<:Integer},
    dataset::MIPVerify.LabelledDataset,
)
    num_samples = MIPVerify.num_samples(dataset)
    @assert(minimum(target_indices) >= 1, "Target sample indexes must be 1 or larger.")
    @assert(
        maximum(target_indices) <= num_samples,
        "Target sample indexes must be no larger than the total number of samples $(num_samples)."
    )
end

function initialize_batch_solve(
    save_path::String,
    nn::NeuralNet,
    pp::MIPVerify.PerturbationFamily,
    norm_order::Real,
)::Tuple{String,String,String,DataFrames.DataFrame}

    results_dir = "run_results"
    summary_file_name = "summary.csv"

    batch_run_parameters = MIPVerify.BatchRunParameters(nn, pp, norm_order)
    main_path = joinpath(save_path, batch_run_parameters |> string)

    main_path |> mkpath_if_not_present
    joinpath(main_path, results_dir) |> mkpath_if_not_present

    summary_file_path = joinpath(main_path, summary_file_name)
    summary_file_path |> create_summary_file_if_not_present

    dt = read_summary_file(summary_file_path)
    return (results_dir, main_path, summary_file_path, dt)
end

function save_to_disk(
    sample_number::Integer,
    main_path::String,
    results_dir::String,
    summary_file_path::String,
    d::Dict,
)
    if haskey(d, :Model)
        r = extract_results_for_save(d)
        results_file_uuid = get_uuid()
        results_file_relative_path = joinpath(results_dir, "$(results_file_uuid).mat")
        results_file_path = joinpath(main_path, results_file_relative_path)
        summary_line = generate_csv_summary_line(sample_number, results_file_relative_path, r)

        r[:SolveStatus] = string(r[:SolveStatus])
        matwrite(results_file_path, Dict(ascii(string(k)) => v for (k, v) in r))
    else
        summary_line = generate_csv_summary_line_optimal(sample_number, d)
    end

    open(summary_file_path, "a") do file
        writedlm(file, [summary_line], ',')
    end
end

function is_proven_infeasible_status(status)::Bool
    return string(status) == "INFEASIBLE"
end

function has_legacy_objective_value(row::DataFrames.DataFrameRow)::Bool
    objective_value = row[:ObjectiveValue]
    return !ismissing(objective_value) && !(objective_value |> isnan)
end

function has_verified_witness(row::DataFrames.DataFrameRow)::Bool
    if :WitnessVerified in propertynames(row) && !ismissing(row[:WitnessVerified])
        witness_available =
            :WitnessAvailable in propertynames(row) &&
            !ismissing(row[:WitnessAvailable]) &&
            row[:WitnessAvailable]
        return witness_available && row[:WitnessVerified]
    end
    return has_legacy_objective_value(row)
end

function should_resolve_ambiguous(row::DataFrames.DataFrameRow)::Bool
    return !is_proven_infeasible_status(row[:SolveStatus]) && !has_verified_witness(row)
end

function is_verdict_only(row::DataFrames.DataFrameRow)::Bool
    return :VerdictOnly in propertynames(row) && !ismissing(row[:VerdictOnly]) && row[:VerdictOnly]
end

function should_refine_insecure(row::DataFrames.DataFrameRow)::Bool
    needs_exact_refinement = is_verdict_only(row) || string(row[:SolveStatus]) != "OPTIMAL"
    return needs_exact_refinement && has_verified_witness(row)
end

function validate_batch_refinement_mode(
    solve_rerun_option::MIPVerify.SolveRerunOption,
    verdict_only::Bool,
)::Nothing
    if solve_rerun_option == MIPVerify.refine_insecure_cases && verdict_only
        throw(
            ArgumentError(
                "refine_insecure_cases requires verdict_only=false because refinement computes an exact objective",
            ),
        )
    end
    return nothing
end

"""
$(SIGNATURES)

Determines whether to run a solve on a sample depending on the `solve_rerun_option` by
looking up information on the most recent completed solve recorded in `summary_dt`
matching `sample_number`.

`summary_dt` is expected to be a `DataFrame` with columns `:SampleNumber`, `:SolveStatus`,
`:ObjectiveValue`, `:WitnessAvailable`, and `:WitnessVerified`. Summaries created before witness
verification use the presence of an objective value as a compatibility fallback.

Behavior for different choices of `solve_rerun_option`:
+ `never`: `true` if and only if there is no previous completed solve.
+ `always`: `true` always.
+ `resolve_ambiguous_cases`: `true` if there is no previous completed solve, or if the
    most recent completed solve has neither a verified counterexample nor a proof of infeasibility.
+ `refine_insecure_cases`: `true` if there is no previous completed solve, or if the most
    recent complete solve a) did find a verified counterexample but b) did not reach a
    provably optimal exact objective. A verdict-only result therefore still needs refinement even
    when its feasibility objective terminated with `OPTIMAL`.
"""
function run_on_sample_for_untargeted_attack(
    sample_number::Integer,
    summary_dt::DataFrame,
    solve_rerun_option::MIPVerify.SolveRerunOption,
)::Bool
    previous_solves = filter(row -> row[:SampleNumber] == sample_number, summary_dt)
    if size(previous_solves)[1] == 0
        return true
    end
    # We now know that previous_solves has at least one element.

    if solve_rerun_option == MIPVerify.never
        return !(sample_number in summary_dt[!, :SampleNumber])
    elseif solve_rerun_option == MIPVerify.always
        return true
    elseif solve_rerun_option == MIPVerify.resolve_ambiguous_cases
        return should_resolve_ambiguous(previous_solves[end, :])
    elseif solve_rerun_option == MIPVerify.refine_insecure_cases
        return should_refine_insecure(previous_solves[end, :])
    else
        throw(DomainError("SolveRerunOption $(solve_rerun_option) unknown."))
    end
end

"""
$(SIGNATURES)

Runs [`find_adversarial_example`](@ref) for the specified neural network `nn` and `dataset`
for samples identified by the `target_indices`, with the target labels for each sample set
to the complement of the true label.

It creates a named directory in `save_path`, with the name summarizing
  1) the name of the network in `nn`,
  2) the perturbation family `pp`,
  3) the `norm_order`.

Within this directory, a summary of all the results is stored in `summary.csv`, and
results from individual runs are stored in the subfolder `run_results`.

This function is designed so that it can be interrupted and restarted cleanly; it relies
on the `summary.csv` file to determine what the results of previous runs are (so modifying
this file manually can lead to unexpected behavior.)

If the summary file already contains a result for a given target index, the
`solve_rerun_option` determines whether we rerun [`find_adversarial_example`](@ref) for this
particular index.

`optimizer` specifies the optimizer used to solve the MIP problem once it has been built
and `main_solve_options` specifies the options that will be passed to the optimizer for the 
main solve.

# Named Arguments:
+ `save_path`: Directory where results will be saved. Defaults to current directory.
+ `pp, norm_order, tightening_algorithm, tightening_options, verdict_only,
  solve_if_predicted_in_targeted` are passed
  through to [`find_adversarial_example`](@ref) and have the same default values;
  see documentation for that function for more details.
+ `solve_rerun_option::MIPVerify.SolveRerunOption`: Options are
  `never`, `always`, `resolve_ambiguous_cases`, and `refine_insecure_cases`.
  See [`run_on_sample_for_untargeted_attack`](@ref) for more details.
  `refine_insecure_cases` requires `verdict_only=false` because it computes an exact objective.
"""
function batch_find_untargeted_attack(
    nn::NeuralNet,
    dataset::MIPVerify.LabelledDataset,
    target_indices::AbstractArray{<:Integer},
    optimizer,
    main_solve_options::Dict;
    save_path::String = ".",
    solve_rerun_option::MIPVerify.SolveRerunOption = MIPVerify.never,
    pp::MIPVerify.PerturbationFamily = MIPVerify.UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    tightening_algorithm::MIPVerify.TighteningAlgorithm = DEFAULT_TIGHTENING_ALGORITHM,
    tightening_options::Dict = MIPVerify.get_default_tightening_options(optimizer),
    solve_if_predicted_in_targeted = true,
    adversarial_example_objective::AdversarialExampleObjective = closest,
    verdict_only::Bool = false,
)::Nothing

    validate_batch_refinement_mode(solve_rerun_option, verdict_only)
    verify_target_indices(target_indices, dataset)
    (results_dir, main_path, summary_file_path, dt) =
        initialize_batch_solve(save_path, nn, pp, norm_order)

    for sample_number in target_indices
        should_run = run_on_sample_for_untargeted_attack(sample_number, dt, solve_rerun_option)
        if should_run
            Memento.info(MIPVerify.LOGGER, "Working on index $(sample_number)")
            input = MIPVerify.get_image(dataset, sample_number)
            true_one_indexed_label = MIPVerify.get_label(dataset, sample_number) + 1
            d = find_adversarial_example(
                nn,
                input,
                true_one_indexed_label,
                optimizer,
                main_solve_options,
                invert_target_selection = true,
                pp = pp,
                norm_order = norm_order,
                tightening_algorithm = tightening_algorithm,
                tightening_options = tightening_options,
                solve_if_predicted_in_targeted = solve_if_predicted_in_targeted,
                adversarial_example_objective = adversarial_example_objective,
                verdict_only = verdict_only,
            )

            save_to_disk(sample_number, main_path, results_dir, summary_file_path, d)
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Determines whether to run a solve on a sample depending on the `solve_rerun_option` by
looking up information on the most recent completed solve recorded in `summary_dt`
matching `sample_number`.

`summary_dt` is expected to be a `DataFrame` with columns `:SampleNumber`, `:TargetIndexes`,
`:SolveStatus`, `:ObjectiveValue`, `:WitnessAvailable`, and `:WitnessVerified`. Summaries created
before witness verification use the presence of an objective value as a compatibility fallback.
"""
function run_on_sample_for_targeted_attack(
    sample_number::Integer,
    target_label::Integer,
    summary_dt::DataFrame,
    solve_rerun_option::MIPVerify.SolveRerunOption,
)::Bool
    match_sample_number = summary_dt[!, :SampleNumber] .== sample_number
    match_target_label = summary_dt[!, :TargetIndexes] .== "[$(target_label)]"
    match = match_sample_number .& match_target_label
    previous_solves = summary_dt[match, :]
    if size(previous_solves)[1] == 0
        return true
    end
    # We now know that previous_solves has at least one element.

    if solve_rerun_option == MIPVerify.never
        return !(sample_number in summary_dt[!, :SampleNumber])
    elseif solve_rerun_option == MIPVerify.always
        return true
    elseif solve_rerun_option == MIPVerify.retarget_infeasible_cases
        last_solve_status = previous_solves[end, :SolveStatus]
        return (last_solve_status == "INFEASIBLE")
    elseif solve_rerun_option == MIPVerify.resolve_ambiguous_cases
        return should_resolve_ambiguous(previous_solves[end, :])
    elseif solve_rerun_option == MIPVerify.refine_insecure_cases
        return should_refine_insecure(previous_solves[end, :])
    else
        throw(DomainError("SolveRerunOption $(solve_rerun_option) unknown."))
    end
end

"""
$(SIGNATURES)

Runs [`find_adversarial_example`](@ref) for the specified neural network `nn` and `dataset`
for samples identified by the `target_indices`, with each of the target labels in `target_labels`
individually targeted.

Otherwise same parameters as [`batch_find_untargeted_attack`](@ref).
"""
function batch_find_targeted_attack(
    nn::NeuralNet,
    dataset::MIPVerify.LabelledDataset,
    target_indices::AbstractArray{<:Integer},
    optimizer,
    main_solve_options::Dict;
    save_path::String = ".",
    solve_rerun_option::MIPVerify.SolveRerunOption = MIPVerify.never,
    target_labels::AbstractArray{<:Integer} = [],
    pp::MIPVerify.PerturbationFamily = MIPVerify.UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    tightening_algorithm::MIPVerify.TighteningAlgorithm = DEFAULT_TIGHTENING_ALGORITHM,
    tightening_options::Dict = MIPVerify.get_default_tightening_options(optimizer),
    solve_if_predicted_in_targeted = true,
    verdict_only::Bool = false,
)::Nothing
    results_dir = "run_results"
    summary_file_name = "summary.csv"

    validate_batch_refinement_mode(solve_rerun_option, verdict_only)
    verify_target_indices(target_indices, dataset)
    (results_dir, main_path, summary_file_path, dt) =
        initialize_batch_solve(save_path, nn, pp, norm_order)

    for sample_number in target_indices
        for target_label in target_labels
            should_run = run_on_sample_for_targeted_attack(
                sample_number,
                target_label,
                dt,
                solve_rerun_option,
            )
            if should_run
                input = MIPVerify.get_image(dataset, sample_number)
                true_one_indexed_label = MIPVerify.get_label(dataset, sample_number) + 1
                if true_one_indexed_label == target_label
                    continue
                end

                Memento.info(
                    MIPVerify.LOGGER,
                    "Working on index $(sample_number), with true_label $(true_one_indexed_label) and target_label $(target_label)",
                )

                d = find_adversarial_example(
                    nn,
                    input,
                    target_label,
                    optimizer,
                    main_solve_options,
                    invert_target_selection = false,
                    pp = pp,
                    norm_order = norm_order,
                    tightening_algorithm = tightening_algorithm,
                    tightening_options = tightening_options,
                    solve_if_predicted_in_targeted = solve_if_predicted_in_targeted,
                    verdict_only = verdict_only,
                )

                save_to_disk(sample_number, main_path, results_dir, summary_file_path, d)
            end
        end
    end
    return nothing
end
