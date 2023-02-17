module TestHelpers

using Test

using JuMP
using MathOptInterface
using TimerOutputs

using MIPVerify
using MIPVerify: find_adversarial_example
using MIPVerify: NeuralNet
using MIPVerify: PerturbationFamily

macro timed_testset(name::String, block)
    # copied from https://github.com/KristofferC/Tensors.jl/blob/master/test/runtests.jl#L8
    return quote
        @timeit "$($(esc(name)))" begin
            @testset "$($(esc(name)))" begin
                $(esc(block))
            end
        end
    end
end

const TEST_DEFAULT_TIGHTENING_ALGORITHM = lp

if Base.find_package("Gurobi") === nothing
    using HiGHS
    optimizer = HiGHS.Optimizer
    main_solve_options = Dict("output_flag" => false)
    tightening_options = Dict("output_flag" => false, "time_limit" => 20.0)
else
    using Gurobi
    env = Gurobi.Env()
    optimizer = () -> Gurobi.Optimizer(env)
    main_solve_options = Dict("OutputFlag" => 0)
    tightening_options = Dict("OutputFlag" => 0, "TimeLimit" => 20)
end

get_optimizer() = optimizer
get_main_solve_options()::Dict = main_solve_options
get_tightening_options()::Dict = tightening_options

function get_new_model()::Model
    return Model(optimizer_with_attributes(get_optimizer(), get_main_solve_options()...))
end

"""
Tests the `find_adversarial_example` function.
  - If `input` is already classified in the target label, `expected_objective_value`
    should be 0.
  - If there is no adversarial example for the specified parameters,
    `expected_objective_value` should be NaN.
  - If there is an adversarial example, checks that the objective value is as expected,
    and that the perturbed output for the target label exceeds the perturbed
    output for all other labels by 0.
"""
function test_find_adversarial_example(
    nn::NeuralNet,
    input::Array{<:Real,N},
    target_selection::Union{Integer,Array{<:Integer,1}},
    pp::PerturbationFamily,
    norm_order::Real,
    expected_objective_value::Real;
    invert_target_selection::Bool = false,
) where {N}
    d = find_adversarial_example(
        nn,
        input,
        target_selection,
        get_optimizer(),
        get_main_solve_options(),
        pp = pp,
        norm_order = norm_order,
        tightening_options = get_tightening_options(),
        tightening_algorithm = TEST_DEFAULT_TIGHTENING_ALGORITHM,
        invert_target_selection = invert_target_selection,
    )
    if isnan(expected_objective_value)
        @test d[:SolveStatus] == MathOptInterface.INFEASIBLE ||
              d[:SolveStatus] == MathOptInterface.INFEASIBLE_OR_UNBOUNDED
    else
        actual_objective_value = JuMP.objective_value(d[:Model])
        if expected_objective_value == 0
            @test isapprox(actual_objective_value, expected_objective_value; atol = 1e-4)
        else
            @test isapprox(actual_objective_value, expected_objective_value; rtol = 5e-5)

            perturbed_output = JuMP.value.(d[:PerturbedInput]) |> nn
            perturbed_target_output =
                maximum(perturbed_output[Bool[i ∈ d[:TargetIndexes] for i in 1:length(d[:Output])]])
            maximum_perturbed_other_output =
                maximum(perturbed_output[Bool[i ∉ d[:TargetIndexes] for i in 1:length(d[:Output])]])
            @test perturbed_target_output / (maximum_perturbed_other_output) ≈ 1 atol = 5e-5
        end
    end
end

"""
Runs tests on the neural net described by `nn` for input `input` and the objective values
indicated in `expected objective values`.

# Arguments
- `test_cases::Array{Tuple}`:
   each element is: ((target_selection, perturbation_parameter, norm_order), expected_objective_value)
   `expected_objective_value` is `NaN` if there is no perturbation that brings the image
   into the target category.
"""
function batch_test_adversarial_example(
    nn::NeuralNet,
    input::Array{<:Real,N},
    test_cases::Array,
) where {N}
    for (test_params, expected_objective_value) in test_cases
        (target_selection, pp, norm_order) = test_params
        @testset "target labels = $target_selection, $(string(pp)) perturbation, norm order = $norm_order" begin
            test_find_adversarial_example(
                nn,
                input,
                target_selection,
                pp,
                norm_order,
                expected_objective_value,
            )
        end
    end
end
end

"""
Generates a pseudorandom array of the specified `dims` with values in [lb, ub]
"""
function gen_array(dims::NTuple{N,Integer}, lb::Real, ub::Real) where {N}
    #! format: off
    rands = [
        0.823, 0.714, 0.970, 0.265, 0.969, 0.105, 0.242, 0.362, 0.061, 0.994,
        0.910, 0.439, 0.769, 0.092, 0.473, 0.530, 0.753, 0.966, 0.168, 0.245,
        0.164, 0.279, 0.799, 0.114, 0.615, 0.666, 0.853, 0.025, 0.856, 0.013,
        0.177, 0.340, 0.846, 0.766, 0.799, 0.506, 0.388, 0.412, 0.479, 0.980,
        0.278, 0.236, 0.539, 0.745, 0.934, 0.583, 0.147, 0.829, 0.007, 0.819,
        0.203, 0.073, 0.155, 0.585, 0.635, 0.252, 0.564, 0.824, 0.337, 0.432,
        0.042, 0.113, 0.783, 0.414, 0.623, 0.218, 0.621, 0.045, 0.088, 0.958,
        0.068, 0.885, 0.718, 0.451, 0.798, 0.387, 0.531, 0.992, 0.716, 0.845,
        0.361, 0.514, 0.172, 0.628, 0.995, 0.964, 0.083, 0.397, 0.504, 0.188,
        0.973, 0.692, 0.493, 0.218, 0.967, 0.498, 0.033, 0.752, 0.033, 0.641,
        0.585, 0.849, 0.347, 0.379, 0.211, 0.582, 0.453, 0.935, 0.355, 0.099,
        0.539, 0.697, 0.360, 0.333, 0.870, 0.127, 0.656, 0.579, 0.110, 0.270,
        0.260, 0.228, 0.168, 0.414, 0.847, 0.133, 0.409, 0.977, 0.882, 0.037,
        0.910, 0.257, 0.737, 0.545, 0.859, 0.770, 0.106, 0.691, 0.841, 0.309,
        0.167, 0.800, 0.722, 0.275, 0.779, 0.061, 0.139, 0.463, 0.798, 0.498,
        0.655, 0.716, 0.652, 0.042, 0.788, 0.278, 0.468, 0.627, 0.385, 0.862,
        0.575, 0.528, 0.711, 0.426, 0.983, 0.218, 0.577, 0.198, 0.820, 0.944,
        0.868, 0.001, 0.430, 0.468, 0.421, 0.976, 0.024, 0.441, 0.994, 0.760,
        0.967, 0.362, 0.408, 0.288, 0.640, 0.096, 0.074, 0.531, 0.011, 0.361,
        0.767, 0.697, 0.913, 0.272, 0.466, 0.115, 0.150, 0.719, 0.507, 0.769,
        0.469, 0.435, 0.569, 0.596, 0.816, 0.266, 0.376, 0.302, 0.297, 0.828,
        0.062, 0.986, 0.452, 0.885, 0.675, 0.679, 0.115, 0.974, 0.622, 0.696,
        0.353, 0.997, 0.153, 0.147, 0.329, 0.223, 0.866, 0.426, 0.234, 0.766,
        0.767, 0.507, 0.860, 0.876, 0.067, 0.962, 0.277, 0.196, 0.151, 0.903,
        0.043, 0.215, 0.910, 0.161, 0.086, 0.585, 0.411, 0.095, 0.551, 0.682,
        0.267, 0.858, 0.310, 0.262, 0.253, 0.613, 0.020, 0.090, 0.295, 0.593,
        0.066, 0.905, 0.455, 0.942, 0.613, 0.908, 0.304, 0.279, 0.886, 0.515,
        0.156, 0.435, 0.649, 0.926, 0.685, 0.719, 0.779, 0.616, 0.844, 0.328,
        0.605, 0.065, 0.238, 0.644, 0.477, 0.201, 0.867, 0.604, 0.402, 0.137,
        0.135, 0.135, 0.823, 0.395, 0.592, 0.636, 0.275, 0.824, 0.837, 0.735,
        0.838, 0.922, 0.369, 0.495, 0.796, 0.881, 0.616, 0.641, 0.236, 0.165,
        0.914, 0.244, 0.916, 0.403, 0.721, 0.119, 0.320, 0.725, 0.619, 0.102,
        0.300, 0.064, 0.502, 0.997, 0.030, 0.207, 0.534, 0.637, 0.434, 0.324,
        0.722, 0.425, 0.557, 0.245, 0.923, 0.873, 0.267, 0.202, 0.625, 0.388,
        0.119, 0.548, 0.973, 0.426, 0.859, 0.350, 0.459, 0.881, 0.340, 0.773,
        0.767, 0.266, 0.537, 0.179, 0.427, 0.197, 0.470, 0.596, 0.993, 0.912,
        0.801, 0.788, 0.242, 0.899, 0.892, 0.727, 0.031, 0.965, 0.251, 0.801,
        0.035, 0.957, 0.238, 0.246, 0.560, 0.222, 0.964, 0.812, 0.886, 0.753,
        0.484, 0.032, 0.714, 0.337, 0.850, 0.110, 0.629, 0.062, 0.086, 0.459,
        0.899, 0.685, 0.106, 0.184, 0.420, 0.043, 0.157, 0.701, 0.165, 0.237,
        0.951, 0.841, 0.017, 0.268, 0.796, 0.929, 0.680, 0.866, 0.579, 0.103,
        0.801, 0.189, 0.449, 0.247, 0.134, 0.134, 0.169, 0.201, 0.646, 0.318,
        0.124, 0.161, 0.493, 0.881, 0.472, 0.507, 0.613, 0.296, 0.114, 0.214,
        0.114, 0.897, 0.280, 0.889, 0.240, 0.641, 0.908, 0.614, 0.677, 0.453,
        0.079, 0.095, 0.119, 0.572, 0.169, 0.360, 0.803, 0.853, 0.250, 0.259,
        0.776, 0.777, 0.050, 0.556, 0.129, 0.395, 0.468, 0.639, 0.635, 0.306,
        0.104, 0.868, 0.602, 0.687, 0.412, 0.983, 0.349, 0.858, 0.483, 0.538,
        0.838, 0.837, 0.052, 0.836, 0.706, 0.179, 0.270, 0.538, 0.785, 0.463,
        0.184, 0.798, 0.306, 0.852, 0.632, 0.723, 0.151, 0.275, 0.740, 0.732,
        0.312, 0.186, 0.245, 0.935, 0.737, 0.940, 0.379, 0.641, 0.037, 0.338,
        0.196, 0.187, 0.242, 0.291, 0.795, 0.850, 0.454, 0.453, 0.653, 0.167,
        0.873, 0.519, 0.415, 0.667, 0.137, 0.660, 0.499, 0.060, 0.958, 0.309,
        0.654, 0.415, 0.850, 0.972, 0.147, 0.818, 0.667, 0.760, 0.089, 0.286,
        0.586, 0.197, 0.879, 0.489, 0.176, 0.381, 0.394, 0.072, 0.234, 0.205,
        0.632, 0.528, 0.399, 0.105, 0.045, 0.861, 0.850, 0.299, 0.985, 0.568,
        0.735, 0.585, 0.139, 0.961, 0.107, 0.318, 0.098, 0.376, 0.655, 0.404,
        0.548, 0.403, 0.419, 0.349, 0.766, 0.360, 0.654, 0.800, 0.782, 0.685,
        0.476, 0.575, 0.579, 0.148, 0.007, 0.611, 0.760, 0.845, 0.206, 0.975,
        0.602, 0.657, 0.055, 0.170, 0.237, 0.869, 0.656, 0.331, 0.423, 0.982,
        0.791, 0.547, 0.477, 0.308, 0.026, 0.631, 0.903, 0.119, 0.034, 0.166,
        0.049, 0.727, 0.617, 0.520, 0.113, 0.055, 0.582, 0.500, 0.050, 0.665,
        0.482, 0.356, 0.105, 0.702, 0.452, 0.752, 0.956, 0.649, 0.429, 0.745,
        0.665, 0.498, 0.087, 0.828, 0.197, 0.565, 0.108, 0.837, 0.696, 0.703,
        0.842, 0.805, 0.133, 0.798, 0.525, 0.342, 0.664, 0.894, 0.259, 0.781,
        0.169, 0.440, 0.444, 0.787, 0.150, 0.029, 0.162, 0.872, 0.494, 0.700,
        0.938, 0.434, 0.484, 0.647, 0.842, 0.808, 0.307, 0.445, 0.948, 0.500,
        0.783, 0.652, 0.737, 0.261, 0.289, 0.350, 0.754, 0.297, 0.796, 0.190,
        0.505, 0.153, 0.187, 0.875, 0.777, 0.823, 0.127, 0.259, 0.793, 0.424,
        0.622, 0.321, 0.431, 0.430, 0.999, 0.489, 0.524, 0.258, 0.410, 0.791,
        0.306, 0.444, 0.986, 0.542, 0.732, 0.776, 0.830, 0.325, 0.729, 0.692,
        0.796, 0.275, 0.446, 0.312, 0.866, 0.969, 0.122, 0.407, 0.836, 0.351,
        0.505, 0.167, 0.016, 0.440, 0.891, 0.816, 0.546, 0.776, 0.914, 0.606,
        0.392, 0.188, 0.225, 0.638, 0.318, 0.869, 0.182, 0.581, 0.014, 0.045,
        0.479, 0.741, 0.415, 0.120, 0.151, 0.753, 0.354, 0.412, 0.853, 0.509,
        0.068, 0.486, 0.745, 0.625, 0.349, 0.305, 0.997, 0.481, 0.021, 0.443,
        0.479, 0.010, 0.607, 0.692, 0.780, 0.827, 0.038, 0.781, 0.493, 0.044,
        0.468, 0.370, 0.779, 0.876, 0.935, 0.428, 0.869, 0.365, 0.557, 0.481,
        0.452, 0.203, 0.117, 0.123, 0.342, 0.573, 0.492, 0.639, 0.832, 0.842,
        0.644, 0.682, 0.952, 0.401, 0.173, 0.016, 0.798, 0.101, 0.571, 0.641,
        0.669, 0.274, 0.930, 0.181, 0.935, 0.072, 0.253, 0.012, 0.836, 0.023,
        0.094, 0.906, 0.276, 0.957, 0.416, 0.934, 0.175, 0.524, 0.975, 0.595,
        0.948, 0.024, 0.931, 0.465, 0.695, 0.220, 0.904, 0.277, 0.598, 0.467,
        0.775, 0.123, 0.215, 0.813, 0.960, 0.119, 0.296, 0.999, 0.431, 0.633,
        0.177, 0.714, 0.040, 0.921, 0.026, 0.746, 0.547, 0.073, 0.189, 0.085,
        0.080, 0.636, 0.225, 0.599, 0.291, 0.444, 0.558, 0.012, 0.039, 0.442,
        0.682, 0.315, 0.490, 0.299, 0.377, 0.529, 0.356, 0.548, 0.166, 0.043,
        0.929, 0.667, 0.433, 0.369, 0.682, 0.436, 0.926, 0.727, 0.411, 0.320,
        0.973, 0.043, 0.718, 0.649, 0.659, 0.347, 0.500, 0.065, 0.815, 0.723,
        0.651, 0.590, 0.191, 0.247, 0.784, 0.244, 0.109, 0.840, 0.106, 0.114,
        0.596, 0.617, 0.117, 0.791, 0.684, 0.770, 0.827, 0.802, 0.085, 0.745,
        0.479, 0.957, 0.538, 0.874, 0.170, 0.798, 0.524, 0.310, 0.865, 0.115,
        0.063, 0.634, 0.708, 0.352, 0.223, 0.127, 0.306, 0.430, 0.808, 0.873,
        0.470, 0.861, 0.315, 0.231, 0.255, 0.584, 0.157, 0.101, 0.181, 0.129,
        0.382, 0.821, 0.872, 0.856, 0.789, 0.669, 0.793, 0.439, 0.371, 0.253,
        0.976, 0.266, 0.095, 0.927, 0.025, 0.250, 0.623, 0.048, 0.458, 0.206,
        0.216, 0.678, 0.090, 0.494, 0.230, 0.376, 0.599, 0.143, 0.948, 0.753,
        0.863, 0.490, 0.401, 0.787, 0.853, 0.410, 0.866, 0.667, 0.311, 0.651,
        0.065, 0.853, 0.235, 0.672, 0.954, 0.753, 0.233, 0.417, 0.547, 0.837,
        0.699, 0.502, 0.072, 0.986, 0.454, 0.973, 0.232, 0.479, 0.799, 0.254,
        0.850, 0.282, 0.799, 0.109, 0.671, 0.595, 0.898, 0.717, 0.535, 0.349,
    ]
    #! format: on
    @assert prod(dims) <= prod(size(rands))
    @assert lb < ub
    xs = reshape(rands[1:prod(dims)], dims)
    return xs * (ub - lb) .+ lb
end
