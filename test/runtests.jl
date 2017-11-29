using MIPVerify: set_log_level
using Base.Test
using Base.Test: @test_throws

@testset "MIPVerify" begin
    MIPVerify.set_log_level("debug")

    include("net_parameters.jl")
    include("layers.jl")
    
    @testset "get_max_index" begin
        @test_throws MethodError get_max_index([])
        @test get_max_index([3]) == 1
        @test get_max_index([3, 1, 4]) == 3
        @test get_max_index([3, 1, 4, 1, 5, 9, 2]) == 6
    end

end

