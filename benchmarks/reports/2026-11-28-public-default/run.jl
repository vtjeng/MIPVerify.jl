# Run from the measured checkout with --project=benchmarks.
# Arguments are the output directory, zero-based lane, and lane count.
using CSV, HiGHS, LinearAlgebra, MIPVerify
include(joinpath(@__DIR__, "..", "..", "BenchmarkHelpers.jl"))
using .BenchmarkHelpers
BLAS.set_num_threads(1)
MIPVerify.set_log_level!("warn")

function verify(nn, input, label, screen)
    return MIPVerify.find_adversarial_example(
        nn,
        input,
        label,
        HiGHS.Optimizer,
        Dict("threads" => 1, "output_flag" => false, "time_limit" => 120.0);
        invert_target_selection = true,
        pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
        norm_order = Inf,
        tightening_algorithm = MIPVerify.target_lp,
        tightening_options = Dict("threads" => 1, "output_flag" => false, "time_limit" => 20.0),
        solve_if_predicted_in_targeted = false,
        collect_stats = true,
        backward_screen = screen,
        adversarial_example_objective = MIPVerify.feasibility,
    )
end

function run(output, lane, lanes; network_name = "WK17a", samples = 2000)
    mkpath(output)
    network = MIPVerify.get_example_network_params("MNIST.$(network_name)_linf0.1_authors")
    dataset = MIPVerify.read_datasets("MNIST").test
    # The shipped weights are Float32. Require screening to accept them before timing.
    preflight =
        network_name == "WK17a" ? ((19, :backward_bound), (63, :corner)) : ((1, :backward_bound),)
    for (image, source) in preflight
        result = verify(
            network,
            MIPVerify.get_image(dataset, image),
            MIPVerify.get_label(dataset, image) + 1,
            true,
        )
        get(result, :VerificationSource, nothing) == source ||
            error("Expected active screen route $source on image $image")
        println("preflight image=$image source=$source")
        flush(stdout)
    end
    # Image 2001 exercises screening, 63 candidate search, and 66 the LP/MIP fallback.
    warmup = @elapsed for screen in (false, true),
        image in (network_name == "WK17a" ? (2001, 63, 66) : (2001, 19, 75))

        verify(
            network,
            MIPVerify.get_image(dataset, image),
            MIPVerify.get_label(dataset, image) + 1,
            screen,
        )
    end
    CSV.write(joinpath(output, "setup-$lane.csv"), [(lane = lane, warmup_seconds = warmup)])
    destinations = Dict(
        (screen, pass) =>
            joinpath(output, "$(screen ? "candidate" : "baseline")-$pass-$lane.csv") for
        screen in (false, true), pass in 1:2
    )
    any(isfile, values(destinations)) && error("Output files already exist")
    for image in (lane+1):lanes:samples
        input, label = MIPVerify.get_image(dataset, image), MIPVerify.get_label(dataset, image) + 1
        for pass in 1:2, screen in (isodd(image + pass) ? (false, true) : (true, false))
            result = verify(network, input, label, screen)
            status = string(get(result, :SolveStatus, "SKIPPED_PREDICTED_IN_TARGETED"))
            available, verified = result[:WitnessAvailable], result[:WitnessVerified]
            point = available ? join(vec(result[:PerturbedInputValue]), ';') : ""
            source =
                haskey(result, :Model) ? "MIP" :
                string(get(result, :VerificationSource, :original_input))
            row = (
                sample_index = image,
                solve_status = status,
                semantic_outcome = classify_semantic_outcome(status, available, verified),
                source = source,
                total_time_seconds = result[:TotalTime],
                main_solve_wall_time_seconds = Float64(get(result, :MainSolveWallTime, 0.0)),
                bound_solver_call_count = Int(get(result, :BoundSolverCallCount, 0)),
                adversarial_example_objective = string(result[:AdversarialExampleObjective]),
                witness_available = available,
                witness_verified = verified,
                witness_target_verified = result[:WitnessTargetVerified],
                witness_perturbation_verified = result[:WitnessPerturbationVerified],
                witness_verification = verified ? "public_api" : "none",
                witness_margin = available ? Float64(result[:WitnessMargin]) : NaN,
                perturbed_input_value = point,
            )
            destination = destinations[(screen, pass)]
            CSV.write(destination, [row]; append = isfile(destination))
        end
        image % 100 <= lanes && println("lane=$lane completed image=$image")
        flush(stdout)
    end
end
run(
    ARGS[1],
    parse(Int, ARGS[2]),
    parse(Int, ARGS[3]);
    network_name = length(ARGS) >= 4 ? ARGS[4] : "WK17a",
    samples = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2000,
)
