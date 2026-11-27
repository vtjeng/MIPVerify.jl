# Time only the checked witness search on images left open by both affine screens.
include(joinpath(@__DIR__, "..", "backward_margin_screen.jl"))
# Supply the parent revision’s benchmark directory as the second argument.
module PGDBaseline
include(joinpath(abspath(ARGS[2]), "backward_margin_screen.jl"))
end
using LinearAlgebra
BLAS.set_num_threads(1)

function search_route(stages, x, y, bounds, corners, corner_first)
    if corner_first
        result = search_checked_witness(stages, x, y, bounds, corners)
        return isnothing(result) ? "solver" : result.source
    end
    proposed = PGDBaseline.find_box_witness(stages, x, y, bounds)
    return isnothing(PGDBaseline.repair_witness(stages, x, proposed, y)) ? "solver" : "ascent"
end

function compare_search(destination)
    cohort = CSV.File(joinpath(@__DIR__, "2026-11-27-affine-order.csv.gz"))
    ds = MIPVerify.read_datasets("MNIST").test
    open(destination, "w") do io
        println(io, "network,image,repeat,corner_first,route,search_seconds")
        for name in ("WK17a", "RSL18a")
            net = MIPVerify.get_example_network_params("MNIST." * name * "_linf0.1_authors")
            stages = compile_stages(net, size(MIPVerify.get_image(ds, 1)))
            warm = true
            for row in cohort
                row.network == name &&
                row.pass == 1 &&
                row.order == "adaptive_first" &&
                !row.certified || continue
                x, y = MIPVerify.get_image(ds, row.image), MIPVerify.get_label(ds, row.image) + 1
                corners = Dict{Int,BitVector}()
                _, bounds, _ = staged_margin_screen(stages, x, y, corners)
                @assert maximum(bounds) >= 0
                if warm
                    for mode in (false, true)
                        search_route(stages, x, y, bounds, corners, mode)
                    end
                    warm = false
                end
                for rep in 1:2
                    modes = isodd(rep + row.image) ? (true, false) : (false, true)
                    for mode in modes
                        seconds = @elapsed route = search_route(stages, x, y, bounds, corners, mode)
                        println(io, join((name, row.image, rep, mode, route, seconds), ','))
                    end
                end
                flush(io)
            end
        end
    end
end

compare_search(ARGS[1])
