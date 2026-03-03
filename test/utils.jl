using Test
@isdefined(TestHelpers) || include("TestHelpers.jl")

TestHelpers.@timed_testset "utils/" begin
    include("utils/import_datasets.jl")
    include("utils/import_example_nets.jl")
    include("utils/import_weights.jl")
    include("utils/prep_data_file.jl")
end
