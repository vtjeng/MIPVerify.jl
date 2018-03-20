@enum SolveRerunOption never=1 always=2 resolve_ambigious_cases=3 refine_insecure_cases=4

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
        r[:TighteningApproach]
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
            "TighteningApproach"
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

Determines whether to run a solve on a sample 
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
    elseif solve_rerun_option == MIPVerify.resolve_ambigious_cases
        last_solve_status = previous_solves[end, :SolveStatus]
        last_objective_value = previous_solves[end, :ObjectiveValue]
        return (last_solve_status == "UserLimit") && (last_objective_value |> isnan)
    elseif solve_rerun_option == MIPVerify.refine_insecure_cases
        last_solve_status = previous_solves[end, :SolveStatus]
        last_objective_value = previous_solves[end, :ObjectiveValue]
        return (last_solve_status == "Optimal") && !(last_objective_value |> isnan)
    else
        throw(DomainError("SolveRerunOption $(solve_rerun_option) unknown."))
    end
end

function get_tightening_approach(
    nn::NeuralNet,
    tightening_algorithm::MIPVerify.TighteningAlgorithm)::String
    string(tightening_algorithm)
end

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
    tightening_algorithm::MIPVerify.TighteningAlgorithm = lp,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = MIPVerify.get_default_tightening_solver(main_solver),
    solve_rerun_option::MIPVerify.SolveRerunOption = MIPVerify.never
    )
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
            d = find_adversarial_example(nn, input, true_one_indexed_label, main_solver, invert_target_selection = true, pp=pp, norm_order=norm_order, tolerance=tolerance, rebuild=rebuild, tightening_algorithm = tightening_algorithm, tightening_solver = tightening_solver)

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
end
