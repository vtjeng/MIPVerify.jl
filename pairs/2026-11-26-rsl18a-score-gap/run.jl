# Run from the repository root using its benchmarks project.
include(abspath("benchmarks/backward_margin_screen.jl"))
using LinearAlgebra
BLAS.set_num_threads(1)
MIPVerify.set_log_level!("warn")

function target_lp_run(nn, x, y)
    d = MIPVerify.find_adversarial_example(
        nn, x, y, HiGHS.Optimizer, Dict("threads"=>1, "output_flag"=>false, "time_limit"=>120.0);
        invert_target_selection=true, pp=MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
        norm_order=Inf, tightening_algorithm=MIPVerify.target_lp,
        tightening_options=Dict("threads"=>1, "output_flag"=>false, "time_limit"=>20.0),
        solve_if_predicted_in_targeted=false, collect_stats=true,
        adversarial_example_objective=MIPVerify.feasibility,
    )
    status = string(get(d, :SolveStatus, "SKIPPED_PREDICTED_IN_TARGETED"))
    return (solve_status=status,
        semantic_outcome=classify_semantic_outcome(status, d[:WitnessAvailable], d[:WitnessVerified]),
        source=haskey(d,:Model) ? "MIP" : "unperturbed", bound_solver_call_count=Int(get(d,:BoundSolverCallCount,0)),
        main_solve_wall_time_seconds=Float64(get(d,:MainSolveWallTime,0.0)),
        witness=d[:WitnessVerified] ? get(d,:PerturbedInputValue,nothing) : nothing)
end

function run_study(dir, lane, stride, last_id, name)
    nn = MIPVerify.get_example_network_params(name)
    ds = MIPVerify.read_datasets("MNIST").test
    xy(i)=(MIPVerify.get_image(ds,i), MIPVerify.get_label(ds,i)+1)
    x,_=xy(2001)
    setup = @elapsed stages=compile_stages(nn,size(x))
    # Check every input coefficient against the native first layer after flattening.
    permuted=nn.layers[1](reshape(collect(1:length(x)),size(x)))
    @assert stages[1].weights[permuted,:] == nn.layers[2].matrix
    @assert stages[1].bias == nn.layers[2].bias
    @assert stages[2].weights == nn.layers[4].matrix && stages[2].bias == nn.layers[4].bias
    run(method,x,y) = method == "target_lp" ? target_lp_run(nn,x,y) : verify_image(nn,stages,x,y)
    warm = @elapsed for idx in (2001,2002)
        x,y=xy(idx)
        @assert last(forward(stages,x)) ≈ nn(x)
        for m in ("target_lp","screen")
            run(m,x,y)
        end
    end
    CSV.write(joinpath(dir,"setup-$lane.csv"), [(;lane, setup, warm)])
    for (n,i) in enumerate(lane+1:stride:last_id)
        println("Image ",i); flush(stdout)
        x,y=xy(i)
        methods = isodd(n+lane) ? ("target_lp","screen","screen","target_lp") : ("screen","target_lp","target_lp","screen")
        for (k,m) in enumerate(methods)
            t=@elapsed r=run(m,x,y)
            file=joinpath(dir,"lane-$lane.csv")
            point=isnothing(r.witness) ? "" : join(vec(r.witness),';')
            row=(image=i, method=m, observation=count(==(m),methods[1:k]), seconds=t,
                source=r.source, status=r.solve_status, outcome=r.semantic_outcome, bound_calls=r.bound_solver_call_count,
                main_seconds=r.main_solve_wall_time_seconds, coordinates=point)
            CSV.write(file,[row]; append=isfile(file))
        end
    end
end
run_study(ARGS[1],parse(Int,ARGS[2]),parse(Int,ARGS[3]),parse(Int,ARGS[4]),ARGS[5])
