# Audit RSL18a hidden ranges and target bounds using the independent exact checker.
# Default: directly certified images must also have strictly negative target bounds.
# --fallback: audit reused ranges without requiring a certificate for every target.
fallback = !isempty(ARGS) && first(ARGS) == "--fallback"
fallback && popfirst!(ARGS)
isempty(ARGS) && error("Supply one-based MNIST image IDs; optionally precede them with --fallback")
all(arg -> tryparse(Int, arg) !== nothing, ARGS) || error("Expected integer image IDs after optional --fallback")
checker = normpath(joinpath(@__DIR__, "..", "2026-11-25-wk17a-backward-bounds", "audit", "check_hidden_bounds.jl"))
source = read(checker, String)
network = "MNIST.WK17a_linf0.1_authors"
comparison = "@assert exact_bound<=Exact(float_bound)"
# Reject upstream drift rather than silently changing an unexpected checker.
length(findall(network, source)) == 1 || error("Expected exactly one network identifier in the archived checker")
length(findall(comparison, source)) == 1 || error("Expected exactly one exact-versus-floating enclosure assertion")
source = replace(source, network => "MNIST.RSL18a_linf0.1_authors")
if !fallback
    source = replace(source, comparison => "@assert exact_bound<=Exact(float_bound)<0")
end
# Fallback mode must retain the enclosure assertion verbatim, with no sign test.
expected_comparison = fallback ? comparison * " (i,j," : "@assert exact_bound<=Exact(float_bound)<0 (i,j,"
length(findall(expected_comparison, source)) == 1 || error("Unexpected enclosure assertion after checker adaptation")
println("mode=", fallback ? "fallback" : "direct-certificate", " images=", length(ARGS))
include_string(Main, source, checker)
