# Adapted from the #219 reproducer. Run from the repository root with --project=test.
using JuMP
using MathOptInterface
using HiGHS
using HiGHS_jll
using MIPVerify
using MIPVerify: BlurringPerturbationFamily, find_adversarial_example, lp

const MOI = MathOptInterface
include(joinpath(pwd(), "test", "TestHelpers.jl"))
using .TestHelpers: gen_array

@assert Base.pkgversion(HiGHS) == v"1.27.0"
@assert Base.pkgversion(HiGHS_jll) == v"1.16.0+0"
native = unsafe_string(HiGHS.Highs_version())
@assert native == "1.16.0"
println("Validation process pid=", getpid(), " OS=", Sys.KERNEL, " arch=", Sys.ARCH,
    " Julia=", VERSION, " HiGHS.jl=", Base.pkgversion(HiGHS),
    " HiGHS_jll=", Base.pkgversion(HiGHS_jll), " native=", native)

function build_nn()
    input = gen_array((1, 6, 6, 1), 0, 1)
    c1_kernel = gen_array((2, 2, 1, 2), -1, 1)
    c1_bias = gen_array((2,), -1, 1)
    l1_kernel = gen_array((18, 4), -1, 1)
    l1_bias = gen_array((4,), -1, 1)
    l2_kernel = gen_array((4, 3), -1, 1)
    l2_bias = gen_array((3,), -1, 1)
    nn = Sequential(
        [
            Conv2d(c1_kernel, c1_bias),
            MaxPool((1, 2, 2, 1)),
            ReLU(),
            Flatten(4),
            Linear(l1_kernel, l1_bias),
            ReLU(),
            Linear(l2_kernel, l2_bias),
        ],
        "probe219",
    )
    return nn, input
end

function solve_case(nn, input; main_options = Dict("output_flag" => false), tightening_extra = Dict())
    topts = convert(
        Dict{String,Any},
        merge(Dict("output_flag" => false, "time_limit" => 20.0), tightening_extra),
    )
    d = find_adversarial_example(
        nn,
        input,
        2,
        HiGHS.Optimizer,
        main_options;
        pp = BlurringPerturbationFamily((5, 5)),
        norm_order = 1,
        tightening_options = topts,
        tightening_algorithm = lp,
    )
    return d
end


function main()
    nn, input = build_nn()
    default_options = Dict{String,Any}("output_flag" => false)
    limited_options = merge(default_options, Dict("time_limit" => 0.01))
    for i in 1:4
        result = solve_case(nn, input; main_options = limited_options)
        status = result[:SolveStatus]
        println("phase=10ms attempt=$i status=$status")
        flush(stdout)
        @assert status == MOI.TIME_LIMIT
    end
    single_thread = Dict{String,Any}("threads" => 1, "parallel" => "off")
    for (label, options) in (("single-thread", single_thread), ("default", Dict{String,Any}()))
        # The scheduler is global. Switch only between solves, with matching tightening options.
        HiGHS.Highs_resetGlobalScheduler(1)
        for i in 1:4
            result = solve_case(nn, input;
                main_options = merge(default_options, options),
                tightening_extra = options)
            status = result[:SolveStatus]
            println("phase=$label attempt=$i status=$status")
            flush(stdout)
            @assert status == MOI.OPTIMAL
            objective = JuMP.objective_value(result[:Model])
            witness = result[:WitnessVerified]
            println("phase=$label attempt=$i objective=$objective witness=$witness")
            flush(stdout)
            @assert isapprox(objective, 6.959316; rtol = 5e-5)
            @assert witness
        end
    end
    println("Completed 4 limited and 8 full solves")
end

main()
