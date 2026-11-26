# Run the repository harness, with identical single-thread settings, a warm-up,
# and per-image progress for both sides. Fresh Julia process per lane and run.
path = abspath("benchmarks/benchmark_wk17a_first100.jl")
source = read(path, String)
source = replace(source, "Dict(\"output_flag\" => false," => "Dict(\"threads\" => 1, \"output_flag\" => false,")
source = replace(source, "for sample_index in sample_indices\n" =>
    "for sample_index in sample_indices\n        println(\"Starting image \",sample_index); flush(stdout)\n")
source = replace(source, r"main\(\)\s*$" => "")
include_string(Main, source, path)
actual = copy(ARGS)
empty!(ARGS)
append!(ARGS, ["--samples","2001","--tightening",actual[4],"--out",actual[6]*"-warmup"])
main()
empty!(ARGS)
append!(ARGS, actual)
main()
