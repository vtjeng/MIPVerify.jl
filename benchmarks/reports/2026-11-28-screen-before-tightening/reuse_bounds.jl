using MIPVerify, HiGHS, LinearAlgebra
BLAS.set_num_threads(1)
MIPVerify.set_log_level!("warn")
nn = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
x = MIPVerify.get_image(MIPVerify.read_datasets("MNIST").test, 480)
# The original comparison checks digit 8. The pruning comparison checks every rival of 9.
all_rivals = length(ARGS) == 2 && ARGS[2] == "all-rivals"
function trial(nn, x, algorithm, all_rivals)
    r = find_adversarial_example(
        nn,
        x,
        all_rivals ? 10 : 9,
        HiGHS.Optimizer,
        Dict("threads"=>1, "output_flag"=>false, "time_limit"=>5.0);
        invert_target_selection = all_rivals,
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
        (
            all_rivals ? (r[:NumBinaryVariables], r[:NumStructuralConstraints]) :
            (r[:ReLUSplitCount],)
        )...,
        r[:SolveStatus],
        r[:WitnessVerified],
    )
end
open(ARGS[1], "w") do io
    counts = all_rivals ? "binaries,constraints" : "split"
    println(
        io,
        "repeat,algorithm,total_seconds,formulation_seconds,bound_solves,$counts,status,verified",
    )
    methods =
        all_rivals ? (MIPVerify.interval_arithmetic, MIPVerify.target_lp) :
        (MIPVerify.lp, MIPVerify.mip)
    # Repeat zero warms each method. Reverse the method order for the first measured repeat.
    for pass in 0:2, alg in (isodd(pass) ? reverse(methods) : methods)
        row = trial(nn, x, alg, all_rivals)
        println(io, join((pass, alg, row...), ","))
        flush(io)
    end
end
