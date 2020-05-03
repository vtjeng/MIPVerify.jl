using Test
using JuMP
using MIPVerify
using MIPVerify: check_size, increment!
@isdefined(TestHelpers) || include("../../TestHelpers.jl")

function test_convolution_layer(
    p::MIPVerify.Conv2d, 
    input::AbstractArray{T, 4}, 
    true_output::AbstractArray{T, 4}
    ) where {T<:Real}
    """
    Tests that passing `input` into the convolution layer `p` produces 
    `true_output`. We test three combinations:

      1) Passing numerical input into a numerical Conv2d layer, verifying
         that we recover the value of `true_output`.
         
      2) Setting up an optimization problem with variables corresponding 
         to the convolution layer and the output (`p_v` and `output_v`).

         `output_v` is constrained to be the result of applying `p_v` to
         `input`, and is also constrained to be equal to `true_output`.

         We verify that, when the optimization problem is solved, applying
         `p_v` to `input` recovers the value of `true_output`.
         
         Note that since the optimization problem is under-determined, we
         cannot assert that `p_v` is equal to `p`.

      3) Setting up an optimization problem with variables corresponding
         to the input and output (`input_v` and `output_v`). 
         
         `output_v` is constrained to be the result of applying `p` to 
         `input_v`, and is also constrained to be equal to `true_output`. 

         We verify that, when the optimization problem is solved, applying
         `p` to `input_v` recovers the value of `true_output`.

         As in case 2), we cannot assert that `input_v` is equal to input.
    """
    input_size = size(input)
    filter_size = size(p.filter)
    bias_size = size(p.bias)
    @testset "Numerical Input, Numerical Layer Parameters" begin
        evaluated_output = MIPVerify.conv2d(input, p)
        @test evaluated_output == true_output
    end
    @testset "Numerical Input, Variable Layer Parameters" begin
        m = TestHelpers.get_new_model()
        filter_v = map(_ -> @variable(m), CartesianIndices(filter_size))
        bias_v = map(_ -> @variable(m), CartesianIndices(bias_size))
        p_v = MIPVerify.Conv2d(filter_v, bias_v, p.stride, p.padding)
        output_v = MIPVerify.conv2d(input, p_v)
        @constraint(m, output_v .== true_output)
        solve(m)

        p_solve = MIPVerify.Conv2d(getvalue(filter_v), getvalue(bias_v), p.stride, p.padding)
        solve_output = MIPVerify.conv2d(input, p_solve)
        @test solve_output≈true_output
    end
    @testset "Variable Input, Numerical Layer Parameters" begin
        m = TestHelpers.get_new_model()
        input_v = map(_ -> @variable(m), CartesianIndices(input_size))
        output_v = MIPVerify.conv2d(input_v, p)
        @constraint(m, output_v .== true_output)
        solve(m)

        solve_output = MIPVerify.conv2d(getvalue(input_v), p)
        @test solve_output≈true_output
    end
end

@testset "conv2d.jl" begin
    @testset "Conv2d" begin
        @testset "Base.show" begin
            filter = ones(3, 3, 2, 5)
            p = Conv2d(filter)
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "Conv2d(2, 5, kernel_size=(3, 3), stride=(1, 1), padding=same)"
        end
        @testset "With Bias" begin
            @testset "Matched Size" begin
                out_channels = 5
                filter = ones(3, 3, 2, out_channels)
                bias = ones(out_channels)
                p = Conv2d(filter, bias)
                @test p.filter == filter
                @test p.bias == bias
            end
            @testset "Unmatched Size" begin
                filter_out_channels = 4
                bias_out_channels = 5
                filter = ones(3, 3, 2, filter_out_channels)
                bias = ones(bias_out_channels)
                @test_throws AssertionError Conv2d(filter, bias)
            end
        end
        @testset "No Bias" begin
            filter = ones(3, 3, 2, 5)
            p = Conv2d(filter)
            @test p.filter == filter
        end
        @testset "JuMP Variables" begin
            m = Model()
            filter_size = (3, 3, 2, 5)
            filter = map(_ -> @variable(m), CartesianIndices(filter_size))
            p = Conv2d(filter)
            @test p.filter == filter
        end
        @testset "check_size" begin
            filter = ones(3, 3, 2, 5)
            p = Conv2d(filter)
            @test check_size(p, (3, 3, 2, 5)) === nothing
            @test_throws AssertionError check_size(p, (3, 3, 2, 4))
        end
    end

    @testset "increment!" begin
        @testset "Real * Real" begin
            @test 7 == increment!(1, 2, 3)
        end
        @testset "JuMP.AffExpr * Real" begin
            m = TestHelpers.get_new_model()
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
        input_size = (1, 4, 4, 2)
        input = reshape(collect(1:prod(input_size)), input_size) .- 16
        filter_size = (3, 3, 2, 1)
        filter = reshape(collect(1:prod(filter_size)), filter_size) .- 9
        bias_size = (1, )
        bias = [1]
        true_output_raw = [
            225  381  405  285;
            502  787  796  532;
            550  823  832  532;
            301  429  417  249;      
        ]
        true_output = reshape(transpose(true_output_raw), (1, 4, 4, 1))
        p = Conv2d(filter, bias)
        test_convolution_layer(p, input, true_output)
    end

    @testset "conv2d with non-unit stride" begin
        input_size = (1, 6, 6, 2)
        input = reshape(collect(1:prod(input_size)), input_size) .- 36
        filter_size = (3, 3, 2, 1)
        filter = reshape(collect(1:prod(filter_size)), filter_size) .- 9
        bias_size = (1, )
        bias = [1]
        stride = 2
        true_output_raw = [
            1597  1615  1120;
            1705  1723  1120;
            903   879   513 ;
        ]
        true_output = reshape(transpose(true_output_raw), (1, 3, 3, 1))
        p = Conv2d(filter, bias, stride)
        test_convolution_layer(p, input, true_output)
    end

    @testset "conv2d with stride 2, odd input shape with even filter shape" begin
        input_size = (1, 5, 5, 2)
        input = reshape(collect(1:prod(input_size)), input_size) .- 25
        filter_size = (4, 4, 2, 1)
        filter = reshape(collect(1:prod(filter_size)), filter_size) .- 16
        bias_size = (1, )
        bias = [1]
        stride = 2
        true_output_raw = [
            1756  2511  1310;
            3065  4097  1969;
            1017  1225  501 ;
        ]
        true_output = reshape(transpose(true_output_raw), (1, 3, 3, 1))
        p = Conv2d(filter, bias, stride)
        test_convolution_layer(p, input, true_output)
    end

    @testset "conv2d with 'valid' padding" begin
        @testset "conv2d with 'valid' padding, odd input and filter size, stride = 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            true_output_raw = [
                63 72 81;
                108 117 126;
                153 162 171
            ]
            true_output = reshape(transpose(true_output_raw), (1, 3, 3, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 'valid' padding, odd input and filter size, stride = 1, channels != 1" begin
            input_size = (1, 5, 5, 2)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 2, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            true_output_raw = [
                351 369 387;
                441 459 477;
                531 549 567
            ]
            true_output = reshape(transpose(true_output_raw), (1, 3, 3, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 'valid' padding, stride = 1, input width != input height" begin
            input_size = (1, 5, 6, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            true_output_raw = [
                63  72  81;
                108 117 126;
                153 162 171;
                198 207 216
            ]
            true_output = reshape(transpose(true_output_raw), (1, 3, 4, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 'valid' padding, stride=1, filter width != filter height" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (2, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            true_output_raw = [
                39  45  51;
                57  69  75;
                81  87  99;
                105 111 117
            ]
            true_output = reshape(transpose(true_output_raw), (1, 4, 3, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 'valid' padding, odd input and filter size, stride != 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 2
            true_output_raw = [
                63  81;
                153 171
            ]
            true_output = reshape(transpose(true_output_raw), (1, 2, 2, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 'valid' padding, odd input size, even filter size, stride = 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (2, 2, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            true_output_raw = [
                16 20 24 28;
                36 40 44 48;
                56 60 64 68;
                76 80 84 88
            ]
            true_output = reshape(transpose(true_output_raw), (1, 4, 4, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 'valid' padding, odd input size, even filter size, stride != 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (2, 2, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 2
            true_output_raw = [
                16 24;
                56 64
            ]
            true_output = reshape(transpose(true_output_raw), (1, 2, 2, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 'valid' padding, even input size, odd filter size, stride = 1" begin
            input_size = (1, 6, 6, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            true_output_raw = [
                72  81  90  99;
                126 135 144 153;
                180 189 198 207;
                234 243 252 261
            ]
            true_output = reshape(transpose(true_output_raw), (1, 4, 4, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 'valid' padding, even input size, odd filter size, stride != 1" begin
            input_size = (1, 6, 6, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 2
            true_output_raw = [
                72  90;
                180 198
            ]
            true_output = reshape(transpose(true_output_raw), (1, 2, 2, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 'valid' padding, even input and filter size, stride = 1" begin
            input_size = (1, 6, 6, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (2, 2, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            true_output_raw = [
                18  22  26  30  34;
                42  46  50  54  58;
                66  70  74  78  82;
                90  94  98  102 106;
                114 118 122 126 130
            ]
            true_output = reshape(transpose(true_output_raw), (1, 5, 5, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 'valid' padding, even input and filter size, stride != 1" begin
            input_size = (1, 6, 6, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (2, 2, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 3
            true_output_raw = [
                18  30;
                90 102
            ]
            true_output = reshape(transpose(true_output_raw), (1, 2, 2, 1))
            p = Conv2d(filter, bias, stride, valid)
            test_convolution_layer(p, input, true_output)
        end
    end

    @testset "conv2d wit fixed padding" begin
        @testset "conv2d with (0, 0) padding, stride = 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            padding = (0, 0)
            true_output_raw = [
                63  72  81;
                108 117 126;
                153 162 171
            ]
            true_output = reshape(transpose(true_output_raw), (1, 3, 3, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (0, 0) padding, stride != 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 2
            padding = (0, 0)
            true_output_raw = [
                63  81;
                153 171;
            ]
            true_output = reshape(transpose(true_output_raw), (1, 2, 2, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (1, 1) padding, stride = 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            padding = (1, 1)
            true_output_raw = [
                16  27  33  39  28;
                39  63  72  81  57;
                69  108 117 126 87;
                99  153 162 171 117;
                76  117 123 129 88
            ]
            true_output = reshape(transpose(true_output_raw), (1, 5, 5, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (1, 1) padding, stride != 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 2
            padding = (1, 1)
            true_output_raw = [
                16 33  28;
                69 117 87;
                76 123 88
            ]
            true_output = reshape(transpose(true_output_raw), (1, 3, 3, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with 1 padding, stride = 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            padding = 1
            true_output_raw = [
                16  27  33  39  28;
                39  63  72  81  57;
                69  108 117 126 87;
                99  153 162 171 117;
                76  117 123 129 88
            ]
            true_output = reshape(transpose(true_output_raw), (1, 5, 5, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (1, 1) padding, input width != input_height, stride = 1" begin
            input_size = (1, 6, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            padding = (1, 1)
            true_output_raw = [
                18  30  36  42  48;
                34  45  72  81  90;
                99  69  81  126 135;
                144 153 105 117 180;
                189 198 207 141 90;
                138 144 150 156 106
            ]
            true_output = reshape(transpose(true_output_raw), (1, 6, 5, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (1, 1) padding, input width != input_height, stride != 1" begin
            input_size = (1, 6, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 2
            padding = (1, 1)
            true_output_raw = [
                18  36  48;
                81  135 153;
                90  144 156
            ]
            true_output = reshape(transpose(true_output_raw), (1, 3, 3, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (1, 2) padding, stride = 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            padding = (1, 2)
            true_output_raw = [
                3   6   9   12  9;
                16  27  33  39  28;
                39  63  72  81  57;
                69  108 117 126 87;
                99  153 162 171 117;
                76  117 123 129 88;
                43  66  69  72  49
            ]
            true_output = reshape(transpose(true_output_raw), (1, 5, 7, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (1, 2) padding, stride != 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 2
            padding = (1, 2)
            true_output_raw = [
               3   9   9;
               39  72  57;
               99  162 117;
               43  69  49
            ]
            true_output = reshape(transpose(true_output_raw), (1, 3, 4, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (1, 1) padding, channels != 1, stride = 1" begin
            input_size = (1, 5, 5, 2)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 2, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            padding = (1, 1)
            true_output_raw = [
                132 204 216 228 156;
                228 351 369 387 264;
                288 441 459 477 324;
                348 531 549 567 384;
                252 384 396 408 276
            ]
            true_output = reshape(transpose(true_output_raw), (1, 5, 5, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (1, 1) padding, channels != 1, stride != 1" begin
            input_size = (1, 5, 5, 2)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 2, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 2
            padding = (1, 1)
            true_output_raw = [
                132 216 156;
                288 459 324;
                252 396 276
            ]
            true_output = reshape(transpose(true_output_raw), (1, 3, 3, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (1, 1, 1, 1) padding, stride = 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            padding = (1, 1, 1, 1)
            true_output_raw = [
                16  27  33  39  28;
                39  63  72  81  57;
                69  108 117 126 87;
                99  153 162 171 117;
                76  117 123 129 88
            ]
            true_output = reshape(transpose(true_output_raw), (1, 5, 5, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end

        @testset "conv2d with (1, 2, 3, 4) padding, stride = 1" begin
            input_size = (1, 5, 5, 1)
            input = reshape([1:prod(input_size);], input_size)
            filter_size = (3, 3, 1, 1)
            filter = ones(filter_size...)
            bias_size = (1, )
            bias = [0]
            stride = 1
            padding = (1, 2, 3, 4)
            true_output_raw = [
               0   0   0   0   0   0;
               3   6   9  12   9   5
               16  27  33  39  28  15;
               39  63  72  81  57  30;
               69  108 117 126 87  45;
               99  153 162 171 117 60;
               76  117 123 129 88  45;
               43  66  69  72  49  25;
               0   0   0   0   0   0;
               0   0   0   0   0   0
            ]
            true_output = reshape(transpose(true_output_raw), (1, 6, 10, 1))
            p = Conv2d(filter, bias, stride, padding)
            test_convolution_layer(p, input, true_output)
        end
    end
end