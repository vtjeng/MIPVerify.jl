using HiGHS, LinearAlgebra, MIPVerify
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

# Time only the public verifier call. Reading the input and writing the row happen outside it.
function measure(io, network, data, id, screen, cycle, pass)
    input = MIPVerify.get_image(data, id)
    label = MIPVerify.get_label(data, id) + 1
    t = @timed verify(network, input, label, screen)
    source = get(t.value, :VerificationSource, :MIP)
    screen && cycle > 0 && @assert source == :backward_bound
    println(
        io,
        join(
            (
                cycle,
                pass,
                id,
                screen,
                source,
                t.value[:TotalTime],
                t.time,
                t.gctime,
                t.bytes,
                t.gcstats.pause,
                t.gcstats.full_sweep,
                t.compile_time,
                t.recompile_time,
            ),
            ',',
        ),
    )
    flush(io)
    return nothing
end
function main()
    println("Julia ", VERSION, "; package ", pathof(MIPVerify))
    flush(stdout)
    net = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    data = MIPVerify.read_datasets("MNIST").test
    open(ARGS[1], "w") do io
        println(
            io,
            "cycle,pass,image,screen,source,api_s,elapsed_s,gc_s,bytes,collections,full_collections,compile_s,recompile_s",
        )
        for screen in (false, true), image in (2001, 63, 66)
            measure(io, net, data, image, screen, 0, 0)
        end
        # Replay the eight selected lane-0 images in their original order.
        for cycle in 1:4, image in (5, 13, 21, 25, 29, 49, 61, 113)
            for pass in 1:2, screen in (isodd(image+pass) ? (false, true) : (true, false))
                measure(io, net, data, image, screen, cycle, pass)
            end
        end
    end
end
main()
