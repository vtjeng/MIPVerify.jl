export batch_find_certificate

@enum SolveRerunOption never=1 always=2 resolve_ambiguous_cases=3 refine_insecure_cases=4

struct BatchRunParameters
    nn::NeuralNet
    pp::PerturbationFamily
    norm_order::Real
    tolerance::Real
end

function Base.show(io::IO, t::BatchRunParameters)
    print(io, 
        "$(t.nn.UUID)_$(t.pp)_norm=$(t.norm_order)_tol=$(t.tolerance)"
    )
end

function mkpath_if_not_present(path::String)
    !isdir(path) ? mkpath(path) : nothing
end

function extract_results_for_save(d::Dict)::Dict
    m = d[:Model]
    r = Dict()
    r[:SolveTime] = getsolvetime(m)
    r[:ObjectiveBound] = getobjbound(m)
    r[:ObjectiveValue] = getobjectivevalue(m)
    r[:TargetIndexes] = d[:TargetIndexes]
    r[:SolveStatus] = d[:SolveStatus]
    r[:PredictedIndex] = d[:PredictedIndex]
    r[:TighteningApproach] = d[:TighteningApproach]
    r[:TotalTime] = d[:TotalTime]
    if !isnan(r[:ObjectiveValue])
        r[:PerturbationValue] = d[:Perturbation] |> getvalue
        r[:PerturbedInputValue] = d[:PerturbedInput] |> getvalue
    end
    return r
end

function is_infeasible(s::Symbol)::Bool
    s == :Infeasible || s == :InfeasibleOrUnbounded
end

function get_uuid()::String
    Dates.format(now(), "yyyy-mm-ddTHH.MM.SS.sss")
end

function generate_csv_summary_line(sample_number::Int, results_file_relative_path::String, r::Dict)
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
        r[:TotalTime]
    ] .|> string
end

function create_summary_file_if_not_present(summary_file_path::String)
    if !isfile(summary_file_path)
        summary_header_line = [
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
            "TotalTime"
        ]

        open(summary_file_path, "w") do file
            writecsv(file, [summary_header_line])
        end
    end
end

function verify_target_sample_numbers(target_sample_numbers::AbstractArray{<:Integer}, dataset::MIPVerify.LabelledDataset)
    num_samples = MIPVerify.num_samples(dataset)
    @assert(
        minimum(target_sample_numbers) >= 1,
        "Target sample indexes must be 1 or larger."
    )
    @assert(
        maximum(target_sample_numbers) <= num_samples,
        "Target sample indexes must be no larger than the total number of samples $(num_samples)."
    )
end

"""
$(SIGNATURES)

Determines whether to run a solve on a sample depending on the `solve_rerun_option` by
looking up information on the most recent completed solve. recorded in `summary_dt`

`summary_dt` is expected to be a `DataFrame` with columns `:SampleNumber`, `:SolveStatus`,
and `:ObjectiveValue`. 

Behavior for different choices of `solve_rerun_option`:
+ `never`: `true` if and only if there is no previous completed solve.
+ `always`: `true` always.
+ `resolve_ambiguous_cases`: `true` if there is no previous completed solve, or if the 
    most recent completed solve a) did not find a counter-example BUT b) the optimization
    was not demosntrated to be infeasible.
+ `refine_insecure_cases`: `true` if there is no previous completed solve, or if the most
    recent complete solve a) did find a counter-example BUT b) we did not reach a 
    provably optimal solution.
"""
function run_on_sample(sample_number::Int, summary_dt::DataFrame, solve_rerun_option::MIPVerify.SolveRerunOption)::Bool
    previous_solves = summary_dt[summary_dt[:SampleNumber].==sample_number, :]
    if size(previous_solves)[1] == 0
        return true
    end
    # We now know that previous_solves has at least one element.

    if solve_rerun_option == MIPVerify.never
        return !(sample_number in summary_dt[:SampleNumber])
    elseif solve_rerun_option == MIPVerify.always
        return true
    elseif solve_rerun_option == MIPVerify.resolve_ambiguous_cases
        last_solve_status = previous_solves[end, :SolveStatus]
        last_objective_value = previous_solves[end, :ObjectiveValue]
        return (last_solve_status == "UserLimit") && (last_objective_value |> isnan)
    elseif solve_rerun_option == MIPVerify.refine_insecure_cases
        last_solve_status = previous_solves[end, :SolveStatus]
        last_objective_value = previous_solves[end, :ObjectiveValue]
        return !(last_solve_status == "Optimal") && !(last_objective_value |> isnan)
    else
        throw(DomainError("SolveRerunOption $(solve_rerun_option) unknown."))
    end
end

function get_tightening_approach(
    nn::NeuralNet,
    tightening_algorithm::MIPVerify.TighteningAlgorithm)::String
    string(tightening_algorithm)
end

"""
$(SIGNATURES)

Runs [`find_adversarial_example`](@ref) for the specified neural network `nn` and `dataset`
for the `target_sample_numbers`, skipping `target_sample_numbers` based on the selected
`solve_rerun_option`.

It creates a named directory in `save_path`, with the name summarizing 
  1) the name of the network in `nn`, 
  2) the perturbation family `pp`, 
  3) the `norm_order`
  4) the `tolerance`.

Within this directory, a summary of all the results is stored in `summary.csv`, and 
results from individual runs are stored in the subfolder `run_results`.

This functioned is designed so that it can be interrupted and restarted cleanly; it relies
on the `summary.csv` file to determine what the results of previous runs are (so modifying
this file manually can lead to unexpected behavior.)

`main_solver` specifies the solver used.

# Named Arguments:
`pp, norm_order, tolerance, rebuild, tightening_algorithm, tightening_solver` are passed
directly to [`find_adversarial_example`](@ref); see that documentation for more details.

+ `pp::PerturbationFamily`: Defaults to `UnrestrictedPerturbationFamily()`. 
+ `norm_order::Real`: Defaults to `1`.
+ `tolerance::Real`: Defaults to `0.0`.
+ `rebuild::Bool`: Defaults to `false`.
+ `tightening_algorithm::MIPVerify.TighteningAlgorithm`: Defaults to `mip`.
+ `tightening_solver`: 
+ `solve_rerun_option::MIPVerify.SolveRerunOption`: Options are 
  `never`, `always`, `resolve_ambiguous_cases`, and `refine_insecure_cases`. 
  See [`run_on_sample`](@ref) for more details.
"""
function batch_find_certificate(
    nn::NeuralNet,
    dataset::MIPVerify.LabelledDataset,
    target_sample_numbers::AbstractArray{<:Integer},
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver;
    save_path::String = ".",
    pp::MIPVerify.PerturbationFamily = MIPVerify.UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    tolerance::Real = 0.0,
    rebuild = false,
    tightening_algorithm::MIPVerify.TighteningAlgorithm = DEFAULT_TIGHTENING_ALGORITHM,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = MIPVerify.get_default_tightening_solver(main_solver),
    solve_rerun_option::MIPVerify.SolveRerunOption = MIPVerify.never,
    cache_model = true
    )::Void
    results_dir = "run_results"
    summary_file_name = "summary.csv"

    batch_run_parameters = MIPVerify.BatchRunParameters(nn, pp, norm_order, tolerance)
    main_path = joinpath(save_path, batch_run_parameters |> string)

    main_path |> mkpath_if_not_present
    joinpath(main_path, results_dir) |> mkpath_if_not_present

    summary_file_path = joinpath(main_path, summary_file_name)
    summary_file_path|> create_summary_file_if_not_present

    verify_target_sample_numbers(target_sample_numbers, dataset)
    
    dt = CSV.read(summary_file_path)

    for sample_number in target_sample_numbers
        if run_on_sample(sample_number, dt, solve_rerun_option)
            # TODO (vtjeng): change images -> input IN DATASET
            # TODO (vtjeng): change function signature for get_image and get_label
            info(MIPVerify.LOGGER, "Working on index $(sample_number)")
            input = MIPVerify.get_image(dataset.images, sample_number)
            true_one_indexed_label = MIPVerify.get_label(dataset.labels, sample_number) + 1
            d = find_adversarial_example(nn, input, true_one_indexed_label, main_solver, invert_target_selection = true, pp=pp, norm_order=norm_order, tolerance=tolerance, rebuild=rebuild, tightening_algorithm = tightening_algorithm, tightening_solver = tightening_solver, cache_model=cache_model)

            r = extract_results_for_save(d)
            results_file_uuid = get_uuid()
            results_file_relative_path = joinpath(results_dir, "$(results_file_uuid).mat")
            results_file_path = joinpath(main_path, results_file_relative_path)

            matwrite(results_file_path, r)

            open(summary_file_path, "a") do file
                summary_line = generate_csv_summary_line(sample_number, results_file_relative_path, r)
                writecsv(file, [summary_line])
            end
        end
    end
    return nothing
end
