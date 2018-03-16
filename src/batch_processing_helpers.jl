struct BatchRunParameters
    nn::NeuralNet
    pp::PerturbationFamily
    norm_order::Real
    tolerance::Real
    tightening_algorithm::TighteningAlgorithm
end

function Base.show(io::IO, t::BatchRunParameters)
    print(io, 
        "$(t.nn.UUID)_$(t.pp)_norm=$(t.norm_order)_tol=$(t.tolerance)_ta=$(t.tightening_algorithm)"
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

function generate_csv_summary_line(sample_index::Int, results_file_relative_path::String, r::Dict)
    [
        sample_index, 
        results_file_relative_path,
        r[:TargetIndexes],
        r[:SolveTime],
        r[:SolveStatus],
        r[:SolveStatus] |> is_infeasible,
        r[:ObjectiveValue],
        r[:ObjectiveBound]
    ] .|> string
end

function create_summary_file_if_not_present(summary_file_path::String)
    if !isfile(summary_file_path)
        summary_header_line = [
            "SampleIndex",
            "ResultRelativePath",
            "TargetIndexes",
            "SolveTime",
            "SolveStatus",
            "IsInfeasible",
            "ObjectiveValue",
            "ObjectiveBound"
        ]

        open(summary_file_path, "w") do file
            writecsv(file, [summary_header_line])
        end
    end
end

function verify_target_sample_indexes(target_sample_indexes::AbstractArray{<:Integer}, dataset::MIPVerify.LabelledDataset)
    num_samples = MIPVerify.num_samples(dataset)
    @assert(
        minimum(target_sample_indexes) >= 1,
        "Target sample indexes must be 1 or larger."
    )
    @assert(
        maximum(target_sample_indexes) <= num_samples,
        "Target sample indexes must be no larger than the total number of samples $(num_samples)."
    )
end

function run_on_index(sample_index::Int, summary_dt::DataFrame, batch_rerun_option::MIPVerify.BatchRerunOption)::Bool
    if batch_rerun_option == MIPVerify.skip_existing
        return !(sample_index in summary_dt[:SampleIndex])
    elseif batch_rerun_option == MIPVerify.redo_existing
        return true
    elseif batch_rerun_option == MIPVerify.give_more_time
        previous_solves = summary_dt[summary_dt[:SampleIndex].==sample_index, :]
        if size(previous_solves)[1] == 0
            return true
        else
            last_solve_status = previous_solves[end, :SolveStatus]
            return last_solve_status == "UserLimit"
        end
    else
        throw(DomainError("BatchRerunOption $(batch_rerun_option) unknown."))
    end
end

function batch_find_certificate(
    nn::NeuralNet,
    dataset::MIPVerify.LabelledDataset,
    target_sample_indexes::AbstractArray{<:Integer},
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver;
    save_path::String = ".",
    pp::MIPVerify.PerturbationFamily = MIPVerify.UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    tolerance::Real = 0.0,
    rebuild = false,
    tightening_algorithm::MIPVerify.TighteningAlgorithm = lp,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = MIPVerify.get_default_tightening_solver(main_solver),
    batch_rerun_option::MIPVerify.BatchRerunOption = MIPVerify.skip_existing
    )
    results_dir = "run_results"
    summary_file_name = "summary.csv"

    batch_run_parameters = MIPVerify.BatchRunParameters(nn, pp, norm_order, tolerance, tightening_algorithm)
    main_path = joinpath(save_path, batch_run_parameters |> string)

    main_path |> mkpath_if_not_present
    joinpath(main_path, results_dir) |> mkpath_if_not_present

    summary_file_path = joinpath(main_path, summary_file_name)
    summary_file_path|> create_summary_file_if_not_present

    verify_target_sample_indexes(target_sample_indexes, dataset)
    
    dt = CSV.read(summary_file_path)
    
    for sample_index in target_sample_indexes
        if run_on_index(sample_index, dt, batch_rerun_option)
            # TODO (vtjeng): change images -> input IN DATASET
            # TODO (vtjeng): change function signature for get_image and get_label
            info(MIPVerify.LOGGER, "Working on index $(sample_index)")
            input = MIPVerify.get_image(dataset.images, sample_index)
            true_one_indexed_label = MIPVerify.get_label(dataset.labels, sample_index) + 1
            d = find_adversarial_example(nn, input, true_one_indexed_label, main_solver, invert_target_selection = true, pp=pp, norm_order=norm_order, tolerance=tolerance, rebuild=rebuild, tightening_algorithm = tightening_algorithm, tightening_solver = tightening_solver)

            r = extract_results_for_save(d)
            results_file_uuid = get_uuid()
            results_file_relative_path = joinpath(results_dir, "$(results_file_uuid).mat")
            results_file_path = joinpath(main_path, results_file_relative_path)

            matwrite(results_file_path, r)

            open(summary_file_path, "a") do file
                summary_line = generate_csv_summary_line(sample_index, results_file_relative_path, r)
                writecsv(file, [summary_line])
            end
        end
    end
end
