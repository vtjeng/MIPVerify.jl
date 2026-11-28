using HiGHS, LinearAlgebra, MIPVerify

function trial(nn, data, image, screen; digit = nothing, margin = 0.125)
    x = MIPVerify.get_image(data, image)
    label = MIPVerify.get_label(data, image) + 1
    target = isnothing(digit) ? mod1(label + 1, 10) : digit + 1
    r = MIPVerify.find_adversarial_example(
        nn,
        x,
        target,
        HiGHS.Optimizer,
        Dict("threads"=>1, "output_flag"=>false, "time_limit"=>5.0);
        adversarial_example_objective = MIPVerify.feasibility,
        pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
        tightening_algorithm = MIPVerify.lp,
        tightening_options = Dict("threads"=>1, "output_flag"=>false, "time_limit"=>2.0),
        margin,
        backward_screen = screen,
    )
    return (
        image,
        target,
        screen,
        r[:TotalTime],
        haskey(r, :Model),
        r[:SolveStatus],
        r[:WitnessVerified],
    )
end
function main()
    BLAS.set_num_threads(1)
    MIPVerify.set_log_level!("warn")
    nn = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    data = MIPVerify.read_datasets("MNIST").test
    for screen in (false, true)
        trial(nn, data, 9, screen)
    end
    stress = length(ARGS) == 2 && ARGS[2] == "fallback"
    open(ARGS[1], "w") do io
        println(io, "repeat,image,target_index,screen,seconds,built_model,status,verified_witness")
        for image in (stress ? (480,) : (1:8)),
            pass in 1:2,
            screen in (isodd(image+pass) ? (false, true) : (true, false))
            # On the known hard 9, ask specifically whether 8 can tie it.
            row=stress ? trial(nn, data, image, screen; digit = 8, margin = 0.0) :
                trial(nn, data, image, screen)
            println(io, join((pass, row...), ','))
            flush(io)
        end
    end
end
main()
