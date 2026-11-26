# Check the unmodified solver points, including points accepted within numerical tolerances.
using CSV, MIPVerify
using MIPVerify: Conv2d, Linear, ReLU, Flatten, Sequential

const Q = Rational{BigInt}
exact(layer::Conv2d) = Conv2d(Q.(layer.filter), Q.(layer.bias), layer.stride, layer.padding)
exact(layer::Linear) = Linear(Q.(layer.matrix), Q.(layer.bias))
exact(layer::Union{ReLU,Flatten}) = layer

function check_points(file)
    network = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    replay = Sequential(exact.(network.layers), "original layers in rational arithmetic")
    data = MIPVerify.read_datasets("MNIST").test
    println("sample_index,pass,in_unit_box,stored_center_radius_ok,byte_center_radius_ok,target_minus_true,matches_inclusive_condition,stored_radius_excess,byte_radius_excess")
    for row in CSV.File(file)
        center = MIPVerify.get_image(data, row.sample_index)
        y = MIPVerify.get_label(data, row.sample_index) + 1
        pixels = reshape(Q.(parse.(Float64, split(row.perturbed_input_value, ';'))), size(center))
        in_unit = all(x -> 0 <= x <= 1, pixels)
        stored_radius = maximum(abs.(pixels .- Q.(center)))
        stored_ok = stored_radius <= 1//10
        bytes = round.(Int, 255 .* center) .// 255
        byte_radius = maximum(abs.(pixels .- bytes))
        bytes_ok = byte_radius <= 1//10
        scores = replay(pixels)
        gap = maximum(scores[setdiff(eachindex(scores), y)]) - scores[y]
        valid = in_unit && stored_ok && bytes_ok && gap >= 0
        println(join((row.sample_index, row.pass, in_unit, stored_ok, bytes_ok, Float64(gap), valid, Float64(max(0, stored_radius - 1//10)), Float64(max(0, byte_radius - 1//10))), ','))
    end
end
check_points(isempty(ARGS) ? joinpath(@__DIR__, "timed-mip-witnesses.csv") : only(ARGS))
