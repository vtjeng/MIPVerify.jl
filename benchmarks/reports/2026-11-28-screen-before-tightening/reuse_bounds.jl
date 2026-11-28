using MIPVerify, HiGHS, LinearAlgebra
BLAS.set_num_threads(1)
MIPVerify.set_log_level!("warn")
nn = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
x = MIPVerify.get_image(MIPVerify.read_datasets("MNIST").test, 480)
# Ask whether digit 8 (class index 9) can tie or beat the other digits on image 480.
function trial(nn, x, algorithm)
    r = find_adversarial_example(
        nn,
        x,
        9,
        HiGHS.Optimizer,
        Dict("threads"=>1, "output_flag"=>false, "time_limit"=>5.0);
        adversarial_example_objective = MIPVerify.feasibility,
        pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
        tightening_algorithm = algorithm,
        tightening_options = Dict("threads"=>1, "output_flag"=>false, "time_limit"=>0.2),
        collect_stats = true,
    )
    return (
        r[:TotalTime],
        r[:FormulationTime],
        r[:BoundSolverCallCount],
        r[:ReLUSplitCount],
        r[:SolveStatus],
        r[:WitnessVerified],
    )
end
open(ARGS[1], "w") do io
    println(
        io,
        "repeat,algorithm,total_seconds,formulation_seconds,bound_solves,split,status,verified",
    )
    # Repeat zero warms each method. Reverse the method order for the first measured repeat.
    for pass in 0:2,
        alg in (isodd(pass) ? (MIPVerify.mip, MIPVerify.lp) : (MIPVerify.lp, MIPVerify.mip))

        row = trial(nn, x, alg)
        println(io, join((pass, alg, row...), ","))
        flush(io)
    end
end
