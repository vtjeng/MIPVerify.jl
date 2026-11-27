# Run with the benchmarks project. Arguments: output CSV, worker number, worker count.
include(joinpath(@__DIR__, "..", "backward_margin_screen.jl"))
using LinearAlgebra
BLAS.set_num_threads(1)
function compare_orders(stages, x, y, order)
    if order == :adaptive_first
        _, bounds, source = staged_margin_screen(stages, x, y)
        return bounds, source == "backward"
    end
    lo=max.(0.0, vec(x) .- 0.1 .- 4eps())
    hi=min.(1.0, vec(x) .+ 0.1 .+ 4eps())
    bounds=fill(Inf, length(stages[end].bias))
    bounds[y]=-Inf
    tighten_forward_margins!(bounds, stages, y, lo, hi)
    maximum(bounds)<0 && return bounds, true
    h, il, ih=hidden_bounds(stages, x)
    for j in findall(>=(0), bounds)
        bounds[j]=min(bounds[j], margin_upper(stages, h, il, ih, y, j))
    end
    return bounds, false
end
function measure(outfile, lane, lanes)
    ds=MIPVerify.read_datasets("MNIST").test
    open(outfile, "w") do io
        println(io, "network,image,pass,order,first_certifies,excluded,certified,seconds")
        for (name, n) in (("WK17a", 2000), ("RSL18a", 100))
            nn=MIPVerify.get_example_network_params("MNIST."*name*"_linf0.1_authors")
            stages=compile_stages(nn, size(MIPVerify.get_image(ds, 1)))
            for mode in (:adaptive_first, :parallel_first)
                compare_orders(
                    stages,
                    MIPVerify.get_image(ds, 2001),
                    MIPVerify.get_label(ds, 2001)+1,
                    mode,
                )
                # Force compilation of the fallback before either timer starts.
                compare_orders(
                    stages,
                    MIPVerify.get_image(ds, 9),
                    MIPVerify.get_label(ds, 9)+1,
                    mode,
                )
            end
            for rep in 1:2, i in lane:lanes:n
                x=MIPVerify.get_image(ds, i)
                y=MIPVerify.get_label(ds, i)+1
                for mode in (
                    isodd(i+rep) ? (:adaptive_first, :parallel_first) :
                    (:parallel_first, :adaptive_first)
                )
                    seconds=@elapsed bounds, first=compare_orders(stages, x, y, mode)
                    println(
                        io,
                        join(
                            (
                                name,
                                i,
                                rep,
                                mode,
                                first,
                                count(<(0), bounds)-1,
                                maximum(bounds)<0,
                                seconds,
                            ),
                            ',',
                        ),
                    )
                end
                flush(io)
            end
        end
    end
end
measure(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]))
