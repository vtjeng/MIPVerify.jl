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

function study(lane, lanes, destination)
    dataset = MIPVerify.read_datasets("MNIST").test
    # Check the signed upper objective on an interval where adaptive and parallel differ.
    for (parallel, expected) in ((false, -1.0), (true, -2/3))
        MIPVerify.BackwardBounds.parallel_lower[] = parallel
        coeff = [-1.0]
        b = MIPVerify.BackwardBounds.backward_relu!(coeff, 0.0, [-1.0], [2.0])
        @assert coeff[1] == expected && iszero(b)
        @assert all(z -> -max(0.0, z) <= coeff[1]*z, -1:0.125:2)
    end
    for (name, n) in (("WK17a", 2000), ("RSL18a", 100))
        network = MIPVerify.get_example_network_params("MNIST.$(name)_linf0.1_authors")
        # Exercise solver fallback and screening before starting the clock.
        for parallel in (false, true), i in (name=="WK17a" ? (19, 63, 66) : (1, 19))
            MIPVerify.BackwardBounds.parallel_lower[] = parallel
            verify(
                network,
                MIPVerify.get_image(dataset, i),
                MIPVerify.get_label(dataset, i)+1,
                true,
            )
        end
        open(joinpath(destination, "$name-$lane.csv"), "w") do io
            println(
                io,
                "network,image,pass,lower,source,status,outcome,verified,total_s,screen_s,mip_s,lp_calls,bounds",
            )
            for i in (lane+1):lanes:n
                x = MIPVerify.get_image(dataset, i)
                y = MIPVerify.get_label(dataset, i)+1
                for rep in 1:2, parallel in (isodd(i+rep) ? (false, true) : (true, false))
                    MIPVerify.BackwardBounds.parallel_lower[] = parallel
                    result = verify(network, x, y, true)
                    status = string(get(result, :SolveStatus, "SKIPPED_PREDICTED_IN_TARGETED"))
                    source =
                        haskey(result, :Model) ? "MIP" :
                        string(get(result, :VerificationSource, :original_input))
                    outcome = classify_semantic_outcome(
                        status,
                        result[:WitnessAvailable],
                        result[:WitnessVerified],
                    )
                    bounds = get(result, :StudyBounds, nothing)
                    @assert source=="original_input" ||
                            (bounds!==nothing && length(bounds)==10 && bounds[y]==-Inf)
                    println(
                        io,
                        join(
                            (
                                name,
                                i,
                                rep,
                                parallel ? "parallel" : "adaptive",
                                source,
                                status,
                                outcome,
                                result[:WitnessVerified],
                                result[:TotalTime],
                                get(result, :StudyScreenSeconds, 0.0),
                                get(result, :MainSolveWallTime, 0.0),
                                get(result, :BoundSolverCallCount, 0),
                                isnothing(bounds) ? "" : join(bounds, ';'),
                            ),
                            ',',
                        ),
                    )
                    flush(io)
                end
            end
        end
    end
end
study(parse(Int, ARGS[1]), parse(Int, ARGS[2]), ARGS[3])
