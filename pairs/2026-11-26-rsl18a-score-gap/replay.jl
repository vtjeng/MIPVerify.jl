# Replay final candidate points using the native model and exact rational arithmetic.
using CSV, MIPVerify
using MIPVerify: Linear, ReLU, Flatten, Sequential
const Q = Rational{BigInt}
function replay(dir)
    nn = MIPVerify.get_example_network_params("MNIST.RSL18a_linf0.1_authors")
    native = Sequential([p isa Linear ? Linear(Q.(p.matrix), Q.(p.bias)) : p for p in nn.layers], "exact RSL18a")
    ds = MIPVerify.read_datasets("MNIST").test
    checked = Dict()
    open(joinpath(dir, "witness-replay.csv"),"w") do io
        println(io, "image,observation,in_box,nonnegative_score_gap,gap")
        for filename in sort(filter(name -> occursin(r"^lane-[0-9]+\.csv$", name), readdir(dir))), row in CSV.File(joinpath(dir, filename))
            row.method == "screen" || continue
            ismissing(row.coordinates) && continue
            result = get!(checked, (row.image, String(row.coordinates))) do
                center = MIPVerify.get_image(ds,row.image)
                pixels = reshape(Q.(parse.(Float64, split(row.coordinates,';'))),size(center))
                label = MIPVerify.get_label(ds,row.image)+1
                exact_centers = round.(Int, 255 .* center) .// 255
                in_box = all(0 .<= pixels .<= 1) && all(abs.(pixels.-exact_centers) .<= 1//10) && all(abs.(pixels.-Q.(center)) .<= 1//10)
                scores = native(pixels)
                gap = maximum(scores[setdiff(eachindex(scores),label)]) - scores[label]
                (in_box, gap >= 0, Float64(gap))
            end
            @assert result[1] && result[2]
            println(io, join((row.image,row.observation,result...),','))
        end
    end
end
replay(only(ARGS))
