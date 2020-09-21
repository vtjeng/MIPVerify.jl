using AutoHashEquals
using MathProgBase
using Serialization

"""
Supertype for types encoding the family of perturbations allowed.
"""
abstract type PerturbationFamily end

struct UnrestrictedPerturbationFamily <: PerturbationFamily end
Base.show(io::IO, pp::UnrestrictedPerturbationFamily) = print(io, "unrestricted")
Base.hash(a::UnrestrictedPerturbationFamily, h::UInt) = hash(:UnrestrictedPerturbationFamily, h)

abstract type RestrictedPerturbationFamily <: PerturbationFamily end

"""
For blurring perturbations, we currently allow colors to "bleed" across color channels -
that is, the value of the output of channel 1 can depend on the input to all channels.
(This is something that is worth reconsidering if we are working on color input).
"""
@auto_hash_equals struct BlurringPerturbationFamily <: RestrictedPerturbationFamily
    blur_kernel_size::NTuple{2}
end
Base.show(io::IO, pp::BlurringPerturbationFamily) =
    print(io, filter(x -> !isspace(x), "blur-$(pp.blur_kernel_size)"))

@auto_hash_equals struct LInfNormBoundedPerturbationFamily <: RestrictedPerturbationFamily
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
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    tightening_algorithm::TighteningAlgorithm,
)::Dict
    notice(
        MIPVerify.LOGGER,
        "Determining upper and lower bounds for the input to each non-linear unit.",
    )
    d = build_reusable_model_uncached(
        nn,
        input,
        pp,
        tightening_solver,
        tightening_algorithm,
    )
    return d
end

function build_reusable_model_uncached(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::UnrestrictedPerturbationFamily,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    tightening_algorithm::TighteningAlgorithm,
)::Dict

    m = Model(solver = tightening_solver)
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm) # TODO: consider writing as seperate function

    input_range = CartesianIndices(size(input))

    # v_input will be constrained to `input` in the caller
    v_input = map(_ -> @variable(m), input_range)
    # v_x0 is the input with the perturbation added
    v_x0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input_range)
    @constraint(m, v_x0 .== v_input + v_e)

    v_output = v_x0 |> nn

    d = Dict(
        :Model => m,
        :PerturbedInput => v_x0,
        :Perturbation => v_x0 - input,
        :Output => v_output,
        :PerturbationFamily => pp,
        :TighteningApproach => string(tightening_algorithm),
    )

    return d
end

function build_reusable_model_uncached(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::BlurringPerturbationFamily,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    tightening_algorithm::TighteningAlgorithm,
)::Dict
    # For blurring perturbations, we build a new model for each input. This enables us to get
    # much better bounds.

    m = Model(solver = tightening_solver)
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)

    input_size = size(input)
    num_channels = size(input)[4]
    filter_size = (pp.blur_kernel_size..., num_channels, num_channels)

    v_f = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), CartesianIndices(filter_size))
    @constraint(m, sum(v_f) == num_channels)
    v_x0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), CartesianIndices(input_size))
    @constraint(m, v_x0 .== input |> Conv2d(v_f))

    v_output = v_x0 |> nn

    d = Dict(
        :Model => m,
        :PerturbedInput => v_x0,
        :Perturbation => v_x0 - input,
        :Output => v_output,
        :BlurKernel => v_f,
        :PerturbationFamily => pp,
        :TighteningApproach => string(tightening_algorithm),
    )

    return d
end

function build_reusable_model_uncached(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::LInfNormBoundedPerturbationFamily,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    tightening_algorithm::TighteningAlgorithm,
)::Dict

    m = Model(solver = tightening_solver)
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)

    input_range = CartesianIndices(size(input))
    # v_e is the perturbation added
    v_e =
        map(_ -> @variable(m, lowerbound = -pp.norm_bound, upperbound = pp.norm_bound), input_range)
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lowerbound = max(0, input[i] - pp.norm_bound),
            upperbound = min(1, input[i] + pp.norm_bound)
        ),
        input_range,
    )
    @constraint(m, v_x0 .== input + v_e)

    v_output = v_x0 |> nn

    d = Dict(
        :Model => m,
        :PerturbedInput => v_x0,
        :Perturbation => v_e,
        :Output => v_output,
        :PerturbationFamily => pp,
        :TighteningApproach => string(tightening_algorithm),
    )

    return d
end

struct MIPVerifyExt
    tightening_algorithm::TighteningAlgorithm
end
