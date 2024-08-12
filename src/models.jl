using Serialization

"""
Supertype for types encoding the family of perturbations allowed.
"""
abstract type PerturbationFamily end

struct UnrestrictedPerturbationFamily <: PerturbationFamily end
Base.show(io::IO, pp::UnrestrictedPerturbationFamily) = print(io, "unrestricted")

abstract type RestrictedPerturbationFamily <: PerturbationFamily end

"""
For blurring perturbations, we currently allow colors to "bleed" across color channels -
that is, the value of the output of channel 1 can depend on the input to all channels.
(This is something that is worth reconsidering if we are working on color input).
"""
struct BlurringPerturbationFamily <: RestrictedPerturbationFamily
    blur_kernel_size::NTuple{2}
end
Base.show(io::IO, pp::BlurringPerturbationFamily) =
    print(io, filter(x -> !isspace(x), "blur-$(pp.blur_kernel_size)"))

struct LInfNormBoundedPerturbationFamily <: RestrictedPerturbationFamily
    norm_bound::Real

    function LInfNormBoundedPerturbationFamily(norm_bound::Real)
        @assert(norm_bound > 0, "Norm bound $(norm_bound) should be positive")
        return new(norm_bound)
    end
end
Base.show(io::IO, pp::LInfNormBoundedPerturbationFamily) =
    print(io, "linf-norm-bounded-$(pp.norm_bound)")

function get_model(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::PerturbationFamily,
    optimizer,
    tightening_options::Dict,
    tightening_algorithm::TighteningAlgorithm,
    activation_patterns::Vector{Vector{Bool}},
)::Dict{Symbol,Any}
    notice(
        MIPVerify.LOGGER,
        "Determining upper and lower bounds for the input to each non-linear unit.",
    )
    m = Model(optimizer_with_attributes(optimizer, tightening_options...))
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)

    # Define variables a_i for each activation pattern
    a_i = @variable(m, [1:length(activation_patterns)], Bin)

    # Constraint: Only one activation pattern can be active
    @constraint(m, sum(a_i) == 1)

    # Create a variable for each neuron to represent the sum of relevant a_j variables
    neuron_count = length(activation_patterns[1])  # Assuming all patterns are of equal length
    neuron_activation_vars = @variable(m, [1:neuron_count])  # Variables for each neuron

    for neuron_idx in 1:neuron_count
        active_patterns = [j for j in 1:length(activation_patterns) if activation_patterns[j][neuron_idx]]

        if !isempty(active_patterns)
            @constraint(m, neuron_activation_vars[neuron_idx] == sum(a_i[j] for j in active_patterns))
        else
            @constraint(m, neuron_activation_vars[neuron_idx] == 0)  # Neuron is never active
        end
    end

    # Store the variables in the model's ext for later use
    m.ext[:activation_patterns] = activation_patterns
    m.ext[:a_i] = a_i
    m.ext[:neuron_activation_vars] = neuron_activation_vars


    d_common = Dict(
        :Model => m,
        :PerturbationFamily => pp,
        :TighteningApproach => string(tightening_algorithm),
    )

    return merge(d_common, get_perturbation_specific_keys(nn, input, pp, m))
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::UnrestrictedPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))

    # v_x0 is the input with the perturbation added
    v_x0 = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_x0 - input, :Output => v_output)
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::BlurringPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}

    input_size = size(input)
    num_channels = size(input)[4]
    filter_size = (pp.blur_kernel_size..., num_channels, num_channels)

    v_f = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), CartesianIndices(filter_size))
    @constraint(m, sum(v_f) == num_channels)
    v_x0 = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), CartesianIndices(input_size))
    @constraint(m, v_x0 .== input |> Conv2d(v_f))

    v_output = v_x0 |> nn

    return Dict(
        :PerturbedInput => v_x0,
        :Perturbation => v_x0 - input,
        :Output => v_output,
        :BlurKernel => v_f,
    )
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::LInfNormBoundedPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))
    # v_e is the perturbation added
    v_e = map(
        _ -> @variable(m, lower_bound = -pp.norm_bound, upper_bound = pp.norm_bound),
        input_range,
    )
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = max(0, input[i] - pp.norm_bound),
            upper_bound = min(1, input[i] + pp.norm_bound)
        ),
        input_range,
    )
    @constraint(m, v_x0 .== input + v_e)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_e, :Output => v_output)
end

struct MIPVerifyExt
    tightening_algorithm::TighteningAlgorithm
end
