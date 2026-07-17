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

"""
    witness_value_in_closed_interval(value, lower, upper; atol, rtol)

Return whether finite `value` lies in the closed interval `[lower, upper]`, allowing the supplied
absolute and relative tolerances at either boundary. Non-finite values always fail.
"""
function witness_value_in_closed_interval(
    value::Real,
    lower::Real,
    upper::Real;
    atol::Real = WITNESS_VERIFICATION_ATOL,
    rtol::Real = WITNESS_VERIFICATION_RTOL,
)::Bool
    isfinite(value) || return false
    satisfies_lower_bound = value >= lower || isapprox(value, lower; atol = atol, rtol = rtol)
    satisfies_upper_bound = value <= upper || isapprox(value, upper; atol = atol, rtol = rtol)
    return satisfies_lower_bound && satisfies_upper_bound
end

"""
    witness_inputs_satisfy_common_constraints(input, perturbed_input)

Return whether the original and perturbed inputs have the same shape, both contain only finite
values, and every perturbed value lies in `[0, 1]`. Boundary comparisons use
`WITNESS_VERIFICATION_ATOL` and `WITNESS_VERIFICATION_RTOL`.
"""
function witness_inputs_satisfy_common_constraints(
    input::Array{<:Real},
    perturbed_input::Array{<:Real},
)::Bool
    size(input) == size(perturbed_input) || return false
    all(isfinite, input) || return false
    return all(value -> witness_value_in_closed_interval(value, 0, 1), perturbed_input)
end

"""
    witness_arrays_are_approximately_equal(lhs, rhs)

Return whether `lhs` and `rhs` have the same shape and each pair of values is approximately equal
under the witness-verification tolerances.
"""
function witness_arrays_are_approximately_equal(
    lhs::AbstractArray{<:Real},
    rhs::AbstractArray{<:Real},
)::Bool
    size(lhs) == size(rhs) || return false
    return all(
        i -> isapprox(
            lhs[i],
            rhs[i];
            atol = WITNESS_VERIFICATION_ATOL,
            rtol = WITNESS_VERIFICATION_RTOL,
        ),
        eachindex(lhs, rhs),
    )
end

"""
    blur_kernel_size_is_valid(pp)

Return whether both dimensions of `pp.blur_kernel_size` are positive integers.
"""
function blur_kernel_size_is_valid(pp::BlurringPerturbationFamily)::Bool
    filter_height, filter_width = pp.blur_kernel_size
    return filter_height isa Integer &&
           filter_width isa Integer &&
           filter_height > 0 &&
           filter_width > 0
end

"""
    verify_perturbation_witness(pp, input, perturbed_input, result)

Independently check that `perturbed_input` belongs to `pp` around `input`. Return
`(verified, auxiliary_values)`, where `auxiliary_values` contains numeric diagnostics that the
caller merges into the result even when `verified` is false. Implementations should treat `result`
as read-only and must check the numeric perturbation constraints instead of trusting solver status.

The generic fallback returns `(false, Dict{Symbol,Any}())`, so custom [`PerturbationFamily`](@ref)
implementations fail closed until they define this method. Built-in methods require matching input
shapes, finite values, and a perturbed input in `[0, 1]`. The L-infinity method also checks the
perturbation radius with zero absolute tolerance and `WITNESS_VERIFICATION_RTOL`. The blur method
checks the numeric kernel's shape, coefficient bounds, channel sum, and reconstructed input; the
channel sum uses `WITNESS_VERIFICATION_ATOL` with zero relative tolerance, while other boundary and
reconstruction checks use both witness-verification tolerances.
"""
function verify_perturbation_witness(
    pp::PerturbationFamily,
    input::Array{<:Real},
    perturbed_input::Array{<:Real},
    result::Dict,
)::Tuple{Bool,Dict{Symbol,Any}}
    return false, Dict{Symbol,Any}()
end

function verify_perturbation_witness(
    pp::UnrestrictedPerturbationFamily,
    input::Array{<:Real},
    perturbed_input::Array{<:Real},
    result::Dict,
)::Tuple{Bool,Dict{Symbol,Any}}
    return witness_inputs_satisfy_common_constraints(input, perturbed_input), Dict{Symbol,Any}()
end

function verify_perturbation_witness(
    pp::LInfNormBoundedPerturbationFamily,
    input::Array{<:Real},
    perturbed_input::Array{<:Real},
    result::Dict,
)::Tuple{Bool,Dict{Symbol,Any}}
    witness_inputs_satisfy_common_constraints(input, perturbed_input) ||
        return false, Dict{Symbol,Any}()

    # Do not use the fixed absolute witness tolerance here: it could be much larger than a small
    # perturbation budget. The relative comparison admits only a scale-dependent rounding error.
    satisfies_norm_bound = all(eachindex(input, perturbed_input)) do i
        difference = abs(perturbed_input[i] - input[i])
        difference <= pp.norm_bound ||
            isapprox(difference, pp.norm_bound; atol = 0, rtol = WITNESS_VERIFICATION_RTOL)
    end
    return satisfies_norm_bound, Dict{Symbol,Any}()
end

"""
    identity_blur_kernel(pp, input)

Return a `Float64` identity blur kernel with shape
`(filter_height, filter_width, input_channels, input_channels)`, aligned with [`Conv2d`](@ref)'s
`SAME` padding for odd or even filter dimensions. Return `nothing` when `input` is not four
dimensional, the filter dimensions are invalid, or the input has no channels.
"""
function identity_blur_kernel(
    pp::BlurringPerturbationFamily,
    input::Array{<:Real},
)::Union{Nothing,Array{Float64,4}}
    ndims(input) == 4 || return nothing
    blur_kernel_size_is_valid(pp) || return nothing
    filter_height, filter_width = pp.blur_kernel_size
    num_channels = size(input, 4)
    num_channels > 0 || return nothing

    kernel = zeros(Float64, filter_height, filter_width, num_channels, num_channels)
    # Conv2d uses TensorFlow SAME padding. These indices select the input pixel at the output
    # location for odd and even kernel dimensions.
    identity_height_index = fld(filter_height - 1, 2) + 1
    identity_width_index = fld(filter_width - 1, 2) + 1
    for channel in 1:num_channels
        kernel[identity_height_index, identity_width_index, channel, channel] = 1.0
    end
    return kernel
end

"""
    numeric_blur_kernel(pp, input, result)

Return the numeric blur kernel available in `result`. Prefer a persisted `:WitnessBlurKernel`, then
try to extract values from the JuMP variables in `:BlurKernel`. If neither key exists, construct an
identity kernel for the original-input fast path. Failed JuMP value extraction returns `nothing`
instead of substituting the identity kernel. This function does not mutate `result` or validate the
kernel; [`verify_perturbation_witness`](@ref) performs those checks.
"""
function numeric_blur_kernel(pp::BlurringPerturbationFamily, input::Array{<:Real}, result::Dict)
    if haskey(result, :WitnessBlurKernel)
        return result[:WitnessBlurKernel]
    elseif haskey(result, :BlurKernel)
        return try
            JuMP.value.(result[:BlurKernel])
        catch
            nothing
        end
    end
    return identity_blur_kernel(pp, input)
end

function verify_perturbation_witness(
    pp::BlurringPerturbationFamily,
    input::Array{<:Real},
    perturbed_input::Array{<:Real},
    result::Dict,
)::Tuple{Bool,Dict{Symbol,Any}}
    witness_inputs_satisfy_common_constraints(input, perturbed_input) ||
        return false, Dict{Symbol,Any}()
    ndims(input) == 4 || return false, Dict{Symbol,Any}()
    blur_kernel_size_is_valid(pp) || return false, Dict{Symbol,Any}()

    kernel = numeric_blur_kernel(pp, input, result)
    kernel isa AbstractArray{<:Real,4} || return false, Dict{Symbol,Any}()
    numeric_kernel = Array(kernel)
    values = Dict{Symbol,Any}(:WitnessBlurKernel => numeric_kernel)
    num_channels = size(input, 4)
    expected_kernel_size = (pp.blur_kernel_size..., num_channels, num_channels)
    size(numeric_kernel) == expected_kernel_size || return false, values
    all(value -> witness_value_in_closed_interval(value, 0, 1), numeric_kernel) ||
        return false, values
    # The required sum grows with the channel count, but the permitted residual should not.
    isapprox(sum(numeric_kernel), num_channels; atol = WITNESS_VERIFICATION_ATOL, rtol = 0) ||
        return false, values

    reconstructed_input = try
        input |> Conv2d(numeric_kernel)
    catch
        return false, values
    end
    verified = witness_arrays_are_approximately_equal(reconstructed_input, perturbed_input)
    return verified, values
end

function get_model(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::PerturbationFamily,
    optimizer,
    tightening_options::Dict,
    tightening_algorithm::TighteningAlgorithm,
    collect_stats::Bool = false,
)::Dict{Symbol,Any}
    notice(
        MIPVerify.LOGGER,
        "Determining upper and lower bounds for the input to each non-linear unit.",
    )
    m = Model(optimizer_with_attributes(optimizer, tightening_options...))
    stats = collect_stats ? VerificationStats() : nothing
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm, stats)

    d_common = Dict(
        :Model => m,
        :PerturbationFamily => pp,
        :TighteningApproach => string(tightening_algorithm),
    )

    # The task-local scope lets bound computations on constant expressions (which have no
    # owner model) find the same statistics object.
    perturbation_specific_keys = with_verification_stats(stats) do
        get_perturbation_specific_keys(nn, input, pp, m)
    end
    return merge(d_common, perturbation_specific_keys)
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
