# Compare the arrays covered by the earlier exact proof with the corrected verifier.
using MIPVerify, CSV, SHA

length(ARGS) == 2 || error("Supply the previous and corrected verifier source paths")
const previous_source = abspath(ARGS[1])
const corrected_source = abspath(ARGS[2])
module PreviousVerifier
    include(Main.previous_source)
end
module CorrectedVerifier
    include(Main.corrected_source)
end

function bound_hash(hidden)
    values = Float64[]
    for (lower, upper) in hidden
        append!(values, lower)
        append!(values, upper)
    end
    return bytes2hex(sha256(reinterpret(UInt8, values)))
end

function compare()
    network = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    data = MIPVerify.read_datasets("MNIST").test
    shape = size(MIPVerify.get_image(data, 1))
    previous = PreviousVerifier.compile_stages(network, shape)
    corrected = CorrectedVerifier.compile_stages(network, shape)
    @assert length(previous) == length(corrected)
    for (before, after) in zip(previous, corrected)
        @assert before.weights == after.weights && before.bias == after.bias
    end
    # These are the 34 fallback images in the earlier exact audit linked by README.md.
    images = [66, 212, 246, 327, 407, 444, 480, 512, 544, 551, 584, 668, 690, 724, 867, 895, 952, 983, 1004, 1193, 1291, 1356, 1365, 1396, 1416, 1441, 1552, 1572, 1642, 1774, 1783, 1829, 1851, 1962]
    println("sample_index,neuron_intervals,previous_hidden_sha256,corrected_hidden_sha256,input_boxes_identical,target_bounds_identical")
    for image in images
        center = MIPVerify.get_image(data, image)
        y = MIPVerify.get_label(data, image) + 1
        h1, lo1, hi1 = PreviousVerifier.hidden_bounds(previous, center)
        h2, lo2, hi2 = CorrectedVerifier.hidden_bounds(corrected, center)
        @assert h1 == h2 && lo1 == lo2 && hi1 == hi2
        b1 = PreviousVerifier.margin_bounds(previous, y, h1, lo1, hi1)
        b2 = CorrectedVerifier.margin_bounds(corrected, y, h2, lo2, hi2)
        @assert b1 == b2
        println(join((image, sum(length(lower) for (lower,_) in h1), bound_hash(h1), bound_hash(h2), true, true), ','))
        flush(stdout)
    end
end
compare()
