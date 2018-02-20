using Base.Test
using Base.Test: @test_throws
using JuMP
using MathProgBase
using MIPVerify: MatrixMultiplicationParameters
using MIPVerify: relu, get_target_indexes, set_max_indexes, get_max_index, matmul, tight_upperbound, tight_lowerbound, abs_ge
isdefined(:TestHelpers) || include("../TestHelpers.jl")
using TestHelpers: get_new_model

function count_binary_variables(m::Model)
    count(x -> x == :Bin, m.colCat)
end

@testset "core_ops.jl" begin
    @testset "maximum" begin
        @testset "Variable Input" begin
            @testset "no variables to maximize over" begin
                xs::Array{JuMP.AbstractJuMPScalar} = []
                @test_throws AssertionError MIPVerify.maximum(xs)
            end
            @testset "single variable to maximize over" begin
                m = get_new_model()
                x1 = @variable(m, lowerbound=0, upperbound=3)
                xmax = MIPVerify.maximum([x1])

                # no binary variables need to be introduced
                @test count_binary_variables(m)==0

                @objective(m, Max, x1)
                solve(m)
                solve_output = getvalue(xmax)
                @test solve_output≈3
            end
            @testset "multiple variables to maximize over" begin
                m = get_new_model()
                x1 = @variable(m, lowerbound=0, upperbound=3)
                x2 = @variable(m, lowerbound=4, upperbound=5)
                x3 = @variable(m, lowerbound=2, upperbound=7)
                x4 = @variable(m, lowerbound=-1, upperbound=1)
                x5 = @variable(m, lowerbound=-3, upperbound=1)
                xmax = MIPVerify.maximum([x1, x2, x3, x4, x5])
                
                # an efficient implementation does not add binary variables for x1, x4 and x5
                @test count_binary_variables(m)<= 2
                
                # elements of the input array are made to take their maximum value
                @objective(m, Max, x1+x2+x3+x4+x5)
                solve(m)
                
                solve_output = getvalue(xmax)
                @test solve_output≈7
            end
        end
    end

    @testset "relu" begin
        @testset "Numerical Input" begin
            @test relu(5)==5
            @test relu(0)==0
            @test relu(-1)==0           
        end
        
        @testset "Variable Input" begin
            @testset "strictly non-negative" begin
                m = get_new_model()
                x = @variable(m, lowerbound=0, upperbound=1)
                x_r = relu(x)
                
                # no binary variables should be introduced
                @test count_binary_variables(m)==0
                
                @objective(m, Max, 2*x_r-x)
                solve(m)
                @test getobjectivevalue(m)≈1
            end
            @testset "strictly non-positive" begin
                m = get_new_model()
                x = @variable(m, lowerbound=-1, upperbound=0)
                x_r = relu(x)

                # no binary variables should be introduced
                @test count_binary_variables(m)==0
                
                @objective(m, Max, 2*x_r-x)
                solve(m)
                @test getobjectivevalue(m)≈1
            end
            @testset "regular" begin
                m = get_new_model()
                x = @variable(m, lowerbound=-1, upperbound=2)
                x_r = relu(x)

                # at most one binary variable to be introduced
                @test count_binary_variables(m)<=1
                
                @objective(m, Max, 2*x_r-x)
                solve(m)
                @test getobjectivevalue(m)≈2
            end
        end
    end

    @testset "abs_ge" begin
        @testset "strictly non-negative" begin
            m = get_new_model()
            x = @variable(m, lowerbound=0, upperbound=1)
            x_a = abs_ge(x)
            
            # no binary variables should be introduced
            @test count_binary_variables(m)==0
            
            @objective(m, Max, 2*x_a-x)
            solve(m)
            @test getobjectivevalue(m)≈1
        end
        @testset "strictly non-positive" begin
            m = get_new_model()
            x = @variable(m, lowerbound=-1, upperbound=0)
            x_a = abs_ge(x)

            # no binary variables should be introduced
            @test count_binary_variables(m)==0

            @objective(m, Max, 2*x_a-x)
            solve(m)
            @test getobjectivevalue(m)≈3
        end
        @testset "regular" begin
            m = get_new_model()
            x = @variable(m, lowerbound=-2, upperbound=2)
            x_a = abs_ge(x)

            # no binary variables should be introduced
            @test count_binary_variables(m)==0

            @objective(m, Max, 2*x_a-x)
            solve(m)
            @test getobjectivevalue(m)≈6
        end
        @testset "abs_ge is not strict" begin
            # in particular we only need to satisfy the property |x_a| > x
            m = get_new_model()
            x = @variable(m, lowerbound=-4, upperbound=2)
            x_a = abs_ge(x)

            # no binary variables should be introduced
            @test count_binary_variables(m)==0

            @objective(m, Min, x_a-x)
            solve(m)
            @test getobjectivevalue(m)≈0
        end
    end

    @testset "get_target_indexes" begin
        @test get_target_indexes(1, 5)==[1]
        @test get_target_indexes(1, 5, invert_target_selection = true)==[2, 3, 4, 5]
        @test get_target_indexes([2, 4], 5)==[2, 4]
        @test get_target_indexes([2, 4], 5, invert_target_selection = true)==[1, 3, 5]
        @test_throws AssertionError get_target_indexes(6, 5)
        @test_throws AssertionError get_target_indexes([1, 6], 5)
    end

    @testset "set_max_indexes" begin
        @testset "single target index" begin
            @testset "vanilla" begin
                m = get_new_model()
                x = @variable(m, [i=1:3])
                @constraint(m, x[2] == 5)
                @constraint(m, x[3] == 1)
                set_max_indexes(x, [1])
                @objective(m, Min, x[1])
                solve(m)
                @test getvalue(x[1])≈5
            end
            @testset "with tolerance" begin
                tolerance = 3
                m = get_new_model()
                x = @variable(m, [i=1:3])
                @constraint(m, x[2] == 5)
                @constraint(m, x[3] == 1)
                set_max_indexes(x, [1], tolerance = tolerance)
                @objective(m, Min, x[1])
                solve(m)
                @test getvalue(x[1])≈5+tolerance
            end
        end
        @testset "multiple target indexes" begin
            @testset "vanilla" begin
                m = get_new_model()
                x = @variable(m, [i=1:3])
                @constraint(m, x[1] == 5)
                setlowerbound(x[2], 0)
                setupperbound(x[2], 10)
                setlowerbound(x[3], -1)
                setupperbound(x[3], 10)
                set_max_indexes(x, [2, 3])
                @objective(m, Min, x[2]+x[3])
                solve(m)
                @test getvalue(x[2])≈5
                @test getvalue(x[3])≈-1
            end
            @testset "with tolerance" begin
                tolerance = 3
                m = get_new_model()
                x = @variable(m, [i=1:3])
                @constraint(m, x[1] == 5)
                setlowerbound(x[2], 0)
                setupperbound(x[2], 10)
                setlowerbound(x[3], -1)
                setupperbound(x[3], 10)
                set_max_indexes(x, [2, 3], tolerance = tolerance)
                @objective(m, Min, x[2]+x[3])
                solve(m)
                @test getvalue(x[2])≈5+tolerance
                @test getvalue(x[3])≈-1
            end
        end
    end

    @testset "Bounds" begin
        m = get_new_model()
        x = @variable(m, [i=1:2], lowerbound = -1, upperbound = 1)
        
        A1 = [1 -0.5; -0.5 1]
        b1 = zeros(2)
        p1 = MatrixMultiplicationParameters(A1, b1)
        # naive bounds on our intermediate activations are [-1, 1]
        # for both, but the extremal values are not simultaneously
        # achievable

        A2 = [1 1].'
        b2 = zeros(1)
        p2 = MatrixMultiplicationParameters(A2, b2)
    
        output = matmul(matmul(x, p1), p2)[1]
        @test tight_upperbound(output)≈1
        @test tight_lowerbound(output)≈-1
    end
end