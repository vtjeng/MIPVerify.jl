# Evaluate original network layers and actual timed coordinates in rational arithmetic.
using CSV, MIPVerify
using MIPVerify: Conv2d, Linear, ReLU, Flatten, Sequential

const Q = Rational{BigInt}
exact(layer::Conv2d) = Conv2d(Q.(layer.filter), Q.(layer.bias), layer.stride, layer.padding)
exact(layer::Linear) = Linear(Q.(layer.matrix), Q.(layer.bias))
exact(layer::Union{ReLU,Flatten}) = layer

function evaluate_point(replay, data, image, encoded)
    center = MIPVerify.get_image(data, image)
    true_label = MIPVerify.get_label(data, image) + 1
    pixels = reshape(Q.(parse.(Float64, split(encoded, ';'))), size(center))
    in_unit = all(value -> 0 <= value <= 1, pixels)
    stored_radius = maximum(abs.(pixels .- Q.(center)))
    bytes = round.(Int, 255 .* center) .// 255
    byte_radius = maximum(abs.(pixels .- bytes))
    scores = replay(pixels)
    difference = maximum(scores[setdiff(eachindex(scores), true_label)]) - scores[true_label]
    return (
        in_unit,
        stored_radius <= 1//10,
        byte_radius <= 1//10,
        Float64(difference),
        in_unit && stored_radius <= 1//10 && byte_radius <= 1//10 && difference >= 0,
        Float64(max(0, stored_radius - 1//10)),
        Float64(max(0, byte_radius - 1//10)),
        string(difference),
    )
end

function check_points(files)
    network = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    replay = Sequential(exact.(network.layers), "original layers in rational arithmetic")
    data = MIPVerify.read_datasets("MNIST").test
    cache = Dict()
    all_valid = true
    println("sample_index,pass,run,route,in_unit_box,stored_center_radius_ok,byte_center_radius_ok,target_minus_true,matches_inclusive_condition,stored_radius_excess,byte_radius_excess,target_minus_true_exact")
    for file in files
        for row in CSV.File(file)
            key = (row.sample_index, String(row.perturbed_input_value))
            result = get!(cache, key) do
                evaluate_point(replay, data, row.sample_index, row.perturbed_input_value)
            end
            all_valid &= result[5]
            println(join((row.sample_index, row.pass, row.run, row.route, result...), ','))
            flush(stdout)
        end
    end
    @assert all_valid "At least one returned point fails the exact input or score condition"
    println(stderr, "Unique exact evaluations: ", length(cache))
end
check_points(isempty(ARGS) ? [joinpath(@__DIR__, "timed-witnesses.csv.gz")] : ARGS)
