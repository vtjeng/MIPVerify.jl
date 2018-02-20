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

abstract type PerturbationParameters end

struct AdditivePerturbationParameters <: PerturbationParameters end
Base.show(io::IO, pp::AdditivePerturbationParameters) = print(io, "additive")
Base.hash(a::AdditivePerturbationParameters, h::UInt) = hash(:AdditivePerturbationParameters, h)

@auto_hash_equals struct BlurPerturbationParameters <: PerturbationParameters
    blur_kernel_size::NTuple{2}
end
Base.show(io::IO, pp::BlurPerturbationParameters) = print(io, "blur.$(pp.blur_kernel_size)")

function get_model(
    nn_params::NeuralNetParameters,
    input::Array{<:Real},
    pp::AdditivePerturbationParameters,
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    model_build_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    rebuild::Bool
    )::Dict
    d = get_reusable_model(nn_params, input, pp, model_build_solver, rebuild)
    setsolver(d[:Model], main_solver)
    @constraint(d[:Model], d[:Input] .== input)
    # NOTE (vtjeng): It is important to set the solver before attempting to add a 
    # constraint, as the saved model may have been saved with a different solver (or 
    # different) environment. Flipping the order of the two leads to test failures.
    return d
end

function get_model(
    nn_params::NeuralNetParameters,
    input::Array{<:Real},
    pp::BlurPerturbationParameters,
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    model_build_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    rebuild::Bool
    )::Dict
    d = get_reusable_model(nn_params, input, pp, model_build_solver, rebuild)
    setsolver(d[:Model], main_solver)
    return d
end

function model_hash(
    nn_params::NeuralNetParameters,
    input::Array{<:Real},
    pp::AdditivePerturbationParameters)::UInt
    input_size = size(input)
    return hash(nn_params, hash(input_size, hash(pp)))
end

function model_hash(
    nn_params::NeuralNetParameters,
    input::Array{<:Real},
    pp::BlurPerturbationParameters)::UInt
    return hash(nn_params, hash(input, hash(pp)))
end

function model_filename(
    nn_params::NeuralNetParameters,
    input::Array{<:Real},
    pp::PerturbationParameters)::String
    hash_val = model_hash(nn_params, input, pp)
    input_size = size(input)
    return "$(nn_params.UUID).$(input_size).$(string(pp)).$(hash_val).jls"
end

function get_reusable_model(
    nn_params::NeuralNetParameters,
    input::Array{<:Real},
    pp::PerturbationParameters,
    model_build_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    rebuild::Bool
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
        Rebuilding model from scratch. This may take some time as we determine upper and lower bounds for the input to each non-linear unit. The model built will be cached and re-used for future solves, unless you explicitly set rebuild=false.""")
        d = build_reusable_model_uncached(nn_params, input, pp, model_build_solver)
        open(model_filepath, "w") do f
            serialize(f, d)
        end
    end
    return d
end

function build_reusable_model_uncached(
    nn_params::NeuralNetParameters,
    input::Array{<:Real},
    pp::AdditivePerturbationParameters,
    model_build_solver::MathProgBase.SolverInterface.AbstractMathProgSolver
    )::Dict

    m = Model(solver = model_build_solver)

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
        :PerturbationParameters => pp
    )
    
    return d
end

function build_reusable_model_uncached(
    nn_params::NeuralNetParameters,
    input::Array{<:Real},
    pp::BlurPerturbationParameters,
    model_build_solver::MathProgBase.SolverInterface.AbstractMathProgSolver
    )::Dict
    # For blurring perturbations, we build a new model for each input. This enables us to get
    # much better bounds.

    m = Model(solver = model_build_solver)

    input_size = size(input)
    filter_size = (pp.blur_kernel_size..., 1, 1)

    v_f = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), CartesianRange(filter_size))
    @constraint(m, sum(v_f) == 1)
    v_x0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), CartesianRange(input_size))
    @constraint(m, v_x0 .== input |> Conv2DParameters(v_f))

    v_output = v_x0 |> nn_params

    d = Dict(
        :Model => m,
        :PerturbedInput => v_x0,
        :Perturbation => v_x0 - input,
        :Output => v_output,
        :BlurKernel => v_f,
        :PerturbationParameters => pp
    )

    return d
end