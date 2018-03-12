using AutoHashEquals
using MathProgBase

const module_tempdir = joinpath(Base.tempdir(), "julia", string(module_name(current_module())))
const model_dir = joinpath(module_tempdir, "models")
if !ispath(model_dir)
    mkpath(model_dir)
end

function remove_cached_models()
    if ispath(model_dir)
        rm(model_dir, recursive=true)
        mkpath(model_dir)
    end
end

"""
Supertype for types encoding the family of perturbations allowed.
"""
abstract type PerturbationFamily end

struct UnrestrictedPerturbationFamily <: PerturbationFamily end
Base.show(io::IO, pp::UnrestrictedPerturbationFamily) = print(io, "unrestricted")
Base.hash(a::UnrestrictedPerturbationFamily, h::UInt) = hash(:UnrestrictedPerturbationFamily, h)

abstract type RestrictedPerturbationFamily <: PerturbationFamily end

@auto_hash_equals struct BlurringPerturbationFamily <: RestrictedPerturbationFamily
    blur_kernel_size::NTuple{2}
end
Base.show(io::IO, pp::BlurringPerturbationFamily) = print(io, "blur.$(pp.blur_kernel_size)")

function get_model(
    nn_params::NeuralNet,
    input::Array{<:Real},
    pp::UnrestrictedPerturbationFamily,
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    rebuild::Bool,
    tightening_algorithm::TighteningAlgorithm
    )::Dict
    d = get_reusable_model(nn_params, input, pp, tightening_solver, rebuild, tightening_algorithm)
    setsolver(d[:Model], main_solver)
    @constraint(d[:Model], d[:Input] .== input)
    delete!(d, :Input)
    # NOTE (vtjeng): It is important to set the solver before attempting to add a 
    # constraint, as the saved model may have been saved with a different solver (or 
    # different) environment. Flipping the order of the two leads to test failures.
    return d
end

function get_model(
    nn_params::NeuralNet,
    input::Array{<:Real},
    pp::RestrictedPerturbationFamily,
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    rebuild::Bool,
    tightening_algorithm::TighteningAlgorithm 
    )::Dict
    d = get_reusable_model(nn_params, input, pp, tightening_solver, rebuild, tightening_algorithm)
    setsolver(d[:Model], main_solver)
    return d
end

"""
$(SIGNATURES)

For `UnrestrictedPerturbationFamily`, the search space is simply [0,1] for each pixel.
The model built can thus be re-used for any input with the same input size.
"""
function model_hash(
    nn_params::NeuralNet,
    input::Array{<:Real},
    pp::UnrestrictedPerturbationFamily)::UInt
    input_size = size(input)
    return hash(nn_params, hash(input_size, hash(pp)))
end

"""
$(SIGNATURES)

For `RestrictedPerturbationFamily`, we take advantage of the restricted input search space
corresponding to each nominal (unperturbed) input by considering only activations to the 
non-linear units which are possible for some input in the restricted search space. This 
reduces solve times, but also means that the model must be rebuilt for each different
nominal input.
"""
function model_hash(
    nn_params::NeuralNet,
    input::Array{<:Real},
    pp::RestrictedPerturbationFamily)::UInt
    return hash(nn_params, hash(input, hash(pp)))
end

function model_filename(
    nn_params::NeuralNet,
    input::Array{<:Real},
    pp::PerturbationFamily)::String
    hash_val = model_hash(nn_params, input, pp)
    input_size = size(input)
    return "$(nn_params.UUID).$(input_size).$(string(pp)).$(hash_val).jls"
end

function get_reusable_model(
    nn_params::NeuralNet,
    input::Array{<:Real},
    pp::PerturbationFamily,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    rebuild::Bool,
    tightening_algorithm::TighteningAlgorithm 
    )::Dict

    filename = model_filename(nn_params, input, pp)
    model_filepath = joinpath(model_dir, filename)

    if isfile(model_filepath) && !rebuild
        notice(MIPVerify.LOGGER, "Loading model from cache.")
        d = open(model_filepath, "r") do f
            deserialize(f)
            # TODO (vtjeng): Catch situations where the saved model has a different name.
        end
    else
        notice(MIPVerify.LOGGER, """
        Rebuilding model from scratch. This may take some time as we determine upper and lower bounds for the input to each non-linear unit. The model built will be cached and re-used for future solves, unless you explicitly set rebuild=true.""")
        d = build_reusable_model_uncached(nn_params, input, pp, tightening_solver, tightening_algorithm)
        open(model_filepath, "w") do f
            serialize(f, d)
        end
    end
    return d
end

function build_reusable_model_uncached(
    nn_params::NeuralNet,
    input::Array{<:Real},
    pp::UnrestrictedPerturbationFamily,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    tightening_algorithm::TighteningAlgorithm 
    )::Dict

    m = Model(solver = tightening_solver)
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)

    input_range = CartesianRange(size(input))

    v_input = map(_ -> @variable(m), input_range) # what you're trying to perturb
    v_e = map(_ -> @variable(m, lowerbound = -1, upperbound = 1), input_range) # perturbation added
    v_x0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), input_range) # perturbation + original image
    @constraint(m, v_x0 .== v_input + v_e)

    v_output = v_x0 |> nn_params

    d = Dict(
        :Model => m,
        :PerturbedInput => v_x0,
        :Perturbation => v_e,
        :Output => v_output,
        :Input => v_input,
        :PerturbationFamily => pp
    )
    
    return d
end

function build_reusable_model_uncached(
    nn_params::NeuralNet,
    input::Array{<:Real},
    pp::BlurringPerturbationFamily,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    tightening_algorithm::TighteningAlgorithm 
    )::Dict
    # For blurring perturbations, we build a new model for each input. This enables us to get
    # much better bounds.

    m = Model(solver = tightening_solver)
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)

    input_size = size(input)
    filter_size = (pp.blur_kernel_size..., 1, 1)

    v_f = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), CartesianRange(filter_size))
    @constraint(m, sum(v_f) == 1)
    v_x0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), CartesianRange(input_size))
    @constraint(m, v_x0 .== input |> Conv2d(v_f))

    v_output = v_x0 |> nn_params

    d = Dict(
        :Model => m,
        :PerturbedInput => v_x0,
        :Perturbation => v_x0 - input,
        :Output => v_output,
        :BlurKernel => v_f,
        :PerturbationFamily => pp
    )

    return d
end

struct MIPVerifyExt
    tightening_algorithm::TighteningAlgorithm
end