using AutoHashEquals
using MathProgBase

const module_tempdir = joinpath(Base.tempdir(), "julia", string(module_name(current_module())))
const model_dir = joinpath(module_tempdir, "models")
if !ispath(model_dir)
    mkpath(model_dir)
end

# TODO: Add support for removing all models.

abstract type PerturbationParameters end

struct AdditivePerturbationParameters <: PerturbationParameters end
Base.string(pp::AdditivePerturbationParameters) = "additive"
Base.hash(a::AdditivePerturbationParameters, h::UInt) = hash(:AdditivePerturbationParameters, h)

@auto_hash_equals struct BlurPerturbationParameters <: PerturbationParameters
    blur_kernel_size::NTuple{2}
end
Base.string(pp::BlurPerturbationParameters) = "blur.$(pp.blur_kernel_size)"

function get_model(
    nn_params::NeuralNetParameters,
    input::Array{T, N},
    pp::AdditivePerturbationParameters,
    solver_type::DataType,
    rebuild::Bool
    )::Dict where {T<:Real, N}
    d = get_reusable_model(nn_params, input, pp, solver_type, rebuild)
    @constraint(d[:Model], d[:Input] .== input)
    setsolver(d[:Model], solver_type())
    return d
end

function get_model(
    nn_params::NeuralNetParameters,
    input::Array{T, N},
    pp::BlurPerturbationParameters,
    solver_type::DataType,
    rebuild::Bool
    )::Dict where {T<:Real, N}
    d = get_reusable_model(nn_params, input, pp, solver_type, rebuild)
    setsolver(d[:Model], solver_type())
    return d
end

function model_hash(
    nn_params::NeuralNetParameters,
    input::Array{T, N},
    pp::AdditivePerturbationParameters)::UInt where {T<:Real, N}
    input_size = size(input)
    return hash(nn_params, hash(input_size, hash(pp)))
end

function model_hash(
    nn_params::NeuralNetParameters,
    input::Array{T, N},
    pp::BlurPerturbationParameters)::UInt where {T<:Real, N}
    return hash(nn_params, hash(input, hash(pp)))
end

function model_filename(
    nn_params::NeuralNetParameters,
    input::Array{T, N},
    pp::PerturbationParameters)::String where {T<:Real, N}
    hash_val = model_hash(nn_params, input, pp)
    input_size = size(input)
    return "$(nn_params.UUID).$(input_size).$(string(pp)).$(hash_val).jls"
end

function get_reusable_model(
    nn_params::NeuralNetParameters,
    input::Array{T, N},
    pp::PerturbationParameters,
    solver_type::DataType,
    rebuild::Bool
    )::Dict where {T<:Real, N}

    filename = model_filename(nn_params, input, pp)
    model_filepath = joinpath(model_dir, filename)
    # TODO: Place in temporary directory.

    if isfile(model_filepath) && !rebuild
        info(get_logger(current_module()), "Loading model from cache.")
        d = open(model_filepath, "r") do f
            deserialize(f)
        end
    else
        info(get_logger(current_module()), "Rebuilding model from scratch.")
        d = build_reusable_model_uncached(nn_params, input, pp, solver_type)
        open(model_filepath, "w") do f
            serialize(f, d)
        end
    end
    return d
end

function build_reusable_model_uncached(
    nn_params::NeuralNetParameters,
    input::Array{T, N},
    pp::AdditivePerturbationParameters,
    solver_type::DataType
    )::Dict where {T<:Real, N}

    s = solver_type()
    MathProgBase.setparameters!(s, Silent = true, TimeLimit = 120)
    m = Model(solver = s)

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
    input::Array{T, N},
    pp::BlurPerturbationParameters,
    solver_type::DataType
    )::Dict where {T<:Real, N}
    # For blurring perturbations, we build a new model for each input. This enables us to get
    # much better bounds.

    s = solver_type()
    MathProgBase.setparameters!(s, Silent = true, TimeLimit = 120)
    m = Model(solver = s)

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