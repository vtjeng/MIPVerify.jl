using Base.Test

@testset "net_components/" begin
    include("core_ops.jl")
    include("layers/main.jl")
    include("nets/main.jl")
end