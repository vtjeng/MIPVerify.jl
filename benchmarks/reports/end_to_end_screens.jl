# Run all three configurations on every assigned image, twice, on the same CPU.
include(joinpath(@__DIR__, "..", "backward_margin_screen.jl"))
using LinearAlgebra
BLAS.set_num_threads(1)
MIPVerify.set_log_level!("warn")

function compare_end_to_end(output, worker, workers)
    configurations = (
        (name = "backward", forward = false, second = false),
        (name = "forward", forward = true, second = false),
        (name = "both_corners", forward = true, second = true),
    )
    test = MIPVerify.read_datasets("MNIST").test
    open(output, "w") do io
        println(
            io,
            "network,image,repeat,method,seconds,source,solve_status,outcome,main_solve_seconds,bound_lp_calls,witness_verification",
        )
        for (name, n) in (("WK17a", 2000), ("RSL18a", 100))
            nn = MIPVerify.get_example_network_params("MNIST." * name * "_linf0.1_authors")
            stages = compile_stages(nn, size(MIPVerify.get_image(test, 1)))
            # Warm the common verifier before starting per-image timers. WK17a
            # image 66 exercises the entire LP/MIP fallback, including validation.
            warm = name == "WK17a" ? 66 : 2001
            verify_image(
                nn,
                stages,
                MIPVerify.get_image(test, warm),
                MIPVerify.get_label(test, warm) + 1,
            )
            for image in worker:workers:n
                x = MIPVerify.get_image(test, image)
                y = MIPVerify.get_label(test, image) + 1
                order = circshift([1, 2, 3], image % 3)
                for rep in 1:2, k in (rep == 1 ? order : reverse(order))
                    mode = configurations[k]
                    seconds = @elapsed result = verify_image(
                        nn,
                        stages,
                        x,
                        y;
                        forward_screen = mode.forward,
                        second_corner = mode.second,
                    )
                    println(
                        io,
                        join(
                            (
                                name,
                                image,
                                rep,
                                mode.name,
                                seconds,
                                result.source,
                                result.solve_status,
                                result.semantic_outcome,
                                result.main_solve_wall_time_seconds,
                                result.bound_solver_call_count,
                                result.witness_verification,
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
compare_end_to_end(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]))
