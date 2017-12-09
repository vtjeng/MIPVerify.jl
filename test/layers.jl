using MIPVerify: increment!, getsliceindex, getpoolview, pool, relu, set_max_index, get_max_index, matmul, tight_upperbound, tight_lowerbound
using Base.Test
using Base.Test: @test_throws

using JuMP
using Cbc
using MathProgBase

@testset "layers/" begin

@testset "conv2d.jl" begin

    @testset "increment!" begin
        @testset "Real * Real" begin
            @test 7 == increment!(1, 2, 3)
        end
        @testset "JuMP.AffExpr * Real" begin
            m = Model()
            x = @variable(m, start=100)
            y = @variable(m, start=1)
            s = 5*x+3*y
            t = 3*x+2*y
            increment!(s, 2, t)
            @test getvalue(s)==1107
            increment!(s, t, -1)
            @test getvalue(s)==805
            increment!(s, x, 3)
            @test getvalue(s)==1105
            increment!(s, y, 2)
            @test getvalue(s)==1107
        end
    end

    @testset "conv2d" begin
        srand(1)
        input_size = (1, 4, 4, 2)
        input = rand(0:5, input_size)
        filter_size = (3, 3, 2, 1)
        filter = rand(0:5, filter_size)
        bias_size = (1, )
        bias = rand(0:5, bias_size)
        true_output_raw = [
            49 74 90 56;
            67 118 140 83;
            66 121 134 80;
            56 109 119 62            
        ]
        true_output = reshape(transpose(true_output_raw), (1, 4, 4, 1))
        p = MIPVerify.Conv2DParameters(filter, bias)
        @testset "Numerical Input, Numerical Layer Parameters" begin
            evaluated_output = MIPVerify.conv2d(input, p)
            @test evaluated_output == true_output
        end
        @testset "Numerical Input, Variable Layer Parameters" begin
            solver = CbcSolver()
            MathProgBase.setparameters!(solver, Silent = true)
            m = Model(solver=solver)
            filter_v = map(_ -> @variable(m), CartesianRange(filter_size))
            bias_v = map(_ -> @variable(m), CartesianRange(bias_size))
            p_v = MIPVerify.Conv2DParameters(filter_v, bias_v)
            output_v = MIPVerify.conv2d(input, p_v)
            @constraint(m, output_v .== true_output)
            solve(m)

            p_solve = MIPVerify.Conv2DParameters(getvalue(filter_v), getvalue(bias_v))
            solve_output = MIPVerify.conv2d(input, p_solve)
            @test solve_output≈true_output
        end
        @testset "Variable Input, Numerical Layer Parameters" begin
            solver = CbcSolver()
            MathProgBase.setparameters!(solver, Silent = true)
            m = Model(solver=solver)
            input_v = map(_ -> @variable(m), CartesianRange(input_size))
            output_v = MIPVerify.conv2d(input_v, p)
            @constraint(m, output_v .== true_output)
            solve(m)

            solve_output = MIPVerify.conv2d(getvalue(input_v), p)
            @test solve_output≈true_output
        end
    end
end

@testset "pool.jl" begin
    @testset "getsliceindex" begin
        @testset "inbounds" begin
            @test getsliceindex(10, 2, 3)==[5, 6]
            @test getsliceindex(10, 3, 4)==[10]
        end
        @testset "out of bounds" begin
            @test getsliceindex(10, 5, 4)==[]
        end
    end
    
    input_size = (6, 6)
    input_array = reshape(1:*(input_size...), input_size)
    @testset "getpoolview" begin
        @testset "inbounds" begin
            @test getpoolview(input_array, (2, 2), (3, 3)) == [29 35; 30 36]
            @test getpoolview(input_array, (1, 1), (3, 3)) == cat(2, [15])
        end
        @testset "out of bounds" begin
            @test length(getpoolview(input_array, (1, 1), (7, 7))) == 0
        end
    end

    @testset "maxpool" begin
        true_output = [
            8 20 32;
            10 22 34;
            12 24 36
        ]
        @testset "Numerical Input" begin
            @test pool(input_array, MaxPoolParameters((2, 2))) == true_output
        end
        @testset "Variable Input" begin
            solver = CbcSolver()
            MathProgBase.setparameters!(solver, Silent = true)
            m = Model(solver=solver)
            input_array_v = map(
                i -> @variable(m, lowerbound=i-2, upperbound=i), 
                input_array
            )
            pool_v = pool(input_array_v, MaxPoolParameters((2, 2)))
            # elements of the input array are made to take their maximum value
            @objective(m, Max, sum(input_array_v))
            solve(m)

            solve_output = getvalue.(pool_v)
            @test solve_output≈true_output
        end
    end

    @testset "avgpool" begin
        @testset "Numerical Input" begin
            true_output = [
                4.5 16.5 28.5;
                6.5 18.5 30.5;
                8.5 20.5 32.5
            ]
            @test pool(input_array, AveragePoolParameters((2, 2))) == true_output
        end
    end

end

@testset "core_ops.jl" begin
    @testset "maximum" begin
        @testset "Variable Input" begin
            solver = CbcSolver()
            MathProgBase.setparameters!(solver, Silent = true)
            m = Model(solver=solver)
            x1 = @variable(m, lowerbound=0, upperbound=3)
            x2 = @variable(m, lowerbound=4, upperbound=5)
            x3 = @variable(m, lowerbound=2, upperbound=7)
            x4 = @variable(m, lowerbound=-1, upperbound=1)
            x5 = @variable(m, lowerbound=-3, upperbound=1)
            xmax = MIPVerify.maximum([x1, x2, x3, x4, x5])
            # elements of the input array are made to take their maximum value
            @objective(m, Max, x1+x2+x3+x4+x5)
            solve(m)
            
            solve_output = getvalue(xmax)
            # an efficient implementation does not add binary variables for x1, x4 and x5
            num_binary_variables = count(x -> x == :Bin, m.colCat)

            @test solve_output≈7
            @test num_binary_variables<= 2
        end
    end

    @testset "relu" begin
        @testset "Numerical Input" begin
            @test relu(5)==5
            @test relu(0)==0
            @test relu(-1)==0           
        end
        
        @testset "Variable Input" begin
            @test 1==1
        end
    end

    @testset "set_max_index" begin
        @testset "no tolerance" begin
            solver = CbcSolver()
            MathProgBase.setparameters!(solver, Silent = true)
            m = Model(solver=solver)
            x = @variable(m, [i=1:3])
            @constraint(m, x[2] == 5)
            @constraint(m, x[3] == 1)
            set_max_index(x, 1)
            @objective(m, Min, x[1])
            solve(m)
            @test getvalue(x[1])≈5
        end
        @testset "with tolerance" begin
            tolerance = 3
            solver = CbcSolver()
            MathProgBase.setparameters!(solver, Silent = true)
            m = Model(solver=solver)
            x = @variable(m, [i=1:3])
            @constraint(m, x[2] == 5)
            @constraint(m, x[3] == 1)
            set_max_index(x, 1, tolerance)
            @objective(m, Min, x[1])
            solve(m)
            @test getvalue(x[1])≈5+tolerance
        end
    end

    @testset "Bounds" begin
        solver = CbcSolver()
        MathProgBase.setparameters!(solver, Silent = true)
        m = Model(solver=solver)
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


end