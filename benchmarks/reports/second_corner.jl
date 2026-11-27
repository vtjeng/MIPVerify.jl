# Compare the two witness searches on the fixed cohort left open by both screens.
include(joinpath(@__DIR__, "..", "backward_margin_screen.jl"))
using LinearAlgebra
BLAS.set_num_threads(1)

function compare_second_corner(output)
    cohort = CSV.File(joinpath(@__DIR__, "2026-11-27-corner-search.csv.gz"))
    ds = MIPVerify.read_datasets("MNIST").test
    open(output, "w") do io
        println(io, "network,image,repeat,second_corner,route,search_seconds")
        for name in ("WK17a", "RSL18a")
            model = MIPVerify.get_example_network_params("MNIST." * name * "_linf0.1_authors")
            stages = compile_stages(model, size(MIPVerify.get_image(ds, 1)))
            warm = true
            for row in cohort
                row.network == name && row.repeat == 1 && row.corner_first || continue
                x = MIPVerify.get_image(ds, row.image)
                y = MIPVerify.get_label(ds, row.image) + 1
                backward, parallel = Dict{Int,BitVector}(), Dict{Int,BitVector}()
                _, bounds, _ = staged_margin_screen(stages, x, y, backward, parallel)
                if warm
                    for directions in (nothing, parallel)
                        search_checked_witness(stages, x, y, bounds, backward, directions)
                    end
                    warm = false
                end
                for rep in 1:2
                    for extra in (isodd(rep + row.image) ? (true, false) : (false, true))
                        seconds = @elapsed result = search_checked_witness(
                            stages,
                            x,
                            y,
                            bounds,
                            backward,
                            extra ? parallel : nothing,
                        )
                        route = isnothing(result) ? "solver" : result.source
                        println(io, join((name, row.image, rep, extra, route, seconds), ','))
                    end
                end
                flush(io)
            end
        end
    end
end
compare_second_corner(only(ARGS))
