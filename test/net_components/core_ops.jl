using Base.Test
using JuMP
using MathProgBase
using MIPVerify
using MIPVerify: relu, get_target_indexes, set_max_indexes, get_max_index, matmul, tight_upperbound, tight_lowerbound, abs_ge, masked_relu
isdefined(:TestHelpers) || include("../TestHelpers.jl")
using TestHelpers: get_new_model

function count_binary_variables(m::Model)
    count(x -> x == :Bin, m.colCat)
end

@testset "core_ops.jl" begin
    @testset "maximum" begin
        @testset "Variable Input" begin
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
            @testset "regression test to deal with indexing issue in v0.8.0" begin
                m = get_new_model()
                x1 = @variable(m, lowerbound=-2, upperbound=2) # upperbound of this variable is low enough that it gets filtered away 
                x2 = @variable(m, lowerbound=2.5, upperbound=100)
                x3 = @variable(m, lowerbound=3, upperbound=3.3)
                xmax = MIPVerify.maximum([x1, x2, x3])
                
                # an efficient implementation does not add binary variables for x1, x4 and x5
                @test count_binary_variables(m)<= 2
                
                # elements of the input array are made to take their maximum value
                @objective(m, Max, xmax)
                solve(m)
                
                solve_output = getvalue(xmax)
                @test solve_output≈100
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

    @testset "masked_relu" begin
        @testset "Numerical Input" begin
            @test masked_relu(5, -1)==0
            @test masked_relu(0, -1)==0
            @test masked_relu(-5, -1)==0
            @test masked_relu(5, 0)==5
            @test masked_relu(0, 0)==0
            @test masked_relu(-5, 0)==0
            @test masked_relu(5, 1)==5
            @test masked_relu(0, 1)==0
            @test masked_relu(-5, 1)==-5           
        end
        
        @testset "Variable Input, single" begin
            @testset "mask is negative" begin
                m = get_new_model()
                x = @variable(m, lowerbound=-1, upperbound=2)
                x_r = masked_relu(x, -1)

                # no binary variables to be introduced
                @test count_binary_variables(m)==0
                
                @objective(m, Max, 2*x_r-x)
                solve(m)
                @test getobjectivevalue(m)≈1
                @test getvalue(x)≈-1
                @test getvalue(x_r)≈0

                @objective(m, Min, 2*x_r-x)
                solve(m)
                @test getobjectivevalue(m)≈-2
                @test getvalue(x)≈2
                @test getvalue(x_r)≈0
            end
            @testset "mask is 0" begin
                m = get_new_model()
                x = @variable(m, lowerbound=-1, upperbound=2)
                x_r = masked_relu(x, 0)

                # at most one binary variable to be introduced
                @test count_binary_variables(m)<=1
                
                @objective(m, Max, 2*x_r-x)
                solve(m)
                @test getobjectivevalue(m)≈2
                @test getvalue(x)≈2
                @test getvalue(x_r)≈2

                @objective(m, Min, 2*x_r-x)
                solve(m)
                @test getobjectivevalue(m)≈0
                @test getvalue(x)≈0
                @test getvalue(x_r)≈0
            end
            @testset "mask is positive" begin
                m = get_new_model()
                x = @variable(m, lowerbound=-1, upperbound=2)
                x_r = masked_relu(x, 1)

                # no binary variables to be introduced
                @test count_binary_variables(m)==0
                
                @objective(m, Max, 2*x_r-x)
                solve(m)
                @test getobjectivevalue(m)≈2
                @test getvalue(x)≈2
                @test getvalue(x_r)≈2

                @objective(m, Min, 2*x_r-x)
                solve(m)
                @test getobjectivevalue(m)≈-1
                @test getvalue(x)≈-1
                @test getvalue(x_r)≈-1
            end
        end

        @testset "Variable Input, array" begin
            @testset "invalid mask" begin
                m = get_new_model()
                @variable(m, x[1:4], lowerbound=-1, upperbound=2)

                @test_throws AssertionError masked_relu(x, [-1, 0, 1])
            end
            @testset "valid mask" begin
                m = get_new_model()
                @variable(m, x[1:3], lowerbound=-1, upperbound=2)

                x_r = masked_relu(x, [-1, 0, 1])

                @objective(m, Max, sum(2*x_r-x))
                solve(m)
                @test getobjectivevalue(m)≈5
                @test getvalue(x)≈[-1, 2, 2]
                @test getvalue(x_r)≈[0, 2, 2]

                @objective(m, Min, sum(2*x_r-x))
                solve(m)
                @test getobjectivevalue(m)≈-3
                @test getvalue(x)≈[2, 0, -1]
                @test getvalue(x_r)≈[0, 0, -1]
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
        b1 = [0, 0]
        p1 = Linear(A1, b1)

        A2 = [1 -1; 1 -1]
        b2 = [0, 0]
        p2 = Linear(A2, b2)
        
        test_cases = [
            (interval_arithmetic, -3.0, 3.0),
            (lp, -2.0, 2.0),
            (mip, -1.5, 1.5)
        ]

        for (algorithm, l, u) in test_cases
            @testset "tightening with $(algorithm)" begin
                m.ext[:MIPVerify] = MIPVerify.MIPVerifyExt(algorithm)
                output = (x |> p1 |> ReLU() |> p2)

                @test tight_upperbound(output[1], tightening_algorithm=algorithm)≈u
                @test tight_lowerbound(output[2], tightening_algorithm=algorithm)≈l
            end
        end

    end
end