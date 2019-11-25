using Test

@timed_testset "generated_weights/" begin
    include("generated_weights/mfc+mfc+softmax.jl")
    include("generated_weights/conv+fc+softmax.jl")
end