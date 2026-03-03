using Test
using MIPVerify:
    convert_conv_filter_from_pytorch,
    convert_conv_filter_from_flux,
    convert_linear_weights_from_pytorch,
    convert_images_from_pytorch,
    convert_images_from_flux
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "import_weights.jl" begin
    @testset "Conversion helpers" begin
        @testset "convert_conv_filter_from_pytorch" begin
            # PyTorch OIHW: (2 out, 3 in, 4 h, 5 w)
            pytorch_filter = reshape(collect(1.0:120.0), (2, 3, 4, 5))
            mipverify_filter = convert_conv_filter_from_pytorch(pytorch_filter)
            @test size(mipverify_filter) == (4, 5, 3, 2)  # HWIO
            # Verify element-wise: pytorch[o, i, h, w] == mipverify[h, w, i, o]
            @test mipverify_filter[3, 4, 2, 1] == pytorch_filter[1, 2, 3, 4]
            @test mipverify_filter[1, 1, 1, 1] == pytorch_filter[1, 1, 1, 1]
            @test mipverify_filter[4, 5, 3, 2] == pytorch_filter[2, 3, 4, 5]
        end

        @testset "convert_conv_filter_from_flux" begin
            # Flux WHIO: (5 w, 4 h, 3 in, 2 out)
            flux_filter = reshape(collect(1.0:120.0), (5, 4, 3, 2))
            mipverify_filter = convert_conv_filter_from_flux(flux_filter)
            @test size(mipverify_filter) == (4, 5, 3, 2)  # HWIO
            # Verify element-wise: flux[w, h, i, o] == mipverify[h, w, i, o]
            @test mipverify_filter[3, 4, 2, 1] == flux_filter[4, 3, 2, 1]
            @test mipverify_filter[1, 1, 1, 1] == flux_filter[1, 1, 1, 1]
            @test mipverify_filter[4, 5, 3, 2] == flux_filter[5, 4, 3, 2]
        end

        @testset "convert_linear_weights_from_pytorch" begin
            # PyTorch: (3 out, 4 in)
            pytorch_matrix = reshape(collect(1.0:12.0), (3, 4))
            mipverify_matrix = convert_linear_weights_from_pytorch(pytorch_matrix)
            @test size(mipverify_matrix) == (4, 3)  # (in, out)
            # Verify element-wise: pytorch[o, i] == mipverify[i, o]
            @test mipverify_matrix[2, 1] == pytorch_matrix[1, 2]
            @test mipverify_matrix[4, 3] == pytorch_matrix[3, 4]
        end

        @testset "convert_images_from_pytorch" begin
            # PyTorch NCHW: (2 samples, 3 channels, 4 h, 5 w)
            pytorch_images = reshape(collect(1.0:120.0), (2, 3, 4, 5))
            mipverify_images = convert_images_from_pytorch(pytorch_images)
            @test size(mipverify_images) == (2, 4, 5, 3)  # NHWC
            # Verify element-wise: pytorch[n, c, h, w] == mipverify[n, h, w, c]
            @test mipverify_images[1, 3, 4, 2] == pytorch_images[1, 2, 3, 4]
            @test mipverify_images[2, 4, 5, 3] == pytorch_images[2, 3, 4, 5]
        end

        @testset "convert_images_from_flux" begin
            # Flux WHCN: (5 w, 4 h, 3 channels, 2 samples)
            flux_images = reshape(collect(1.0:120.0), (5, 4, 3, 2))
            mipverify_images = convert_images_from_flux(flux_images)
            @test size(mipverify_images) == (2, 4, 5, 3)  # NHWC
            # Verify element-wise: flux[w, h, c, n] == mipverify[n, h, w, c]
            @test mipverify_images[1, 3, 4, 2] == flux_images[4, 3, 2, 1]
            @test mipverify_images[2, 4, 5, 3] == flux_images[5, 4, 3, 2]
        end
    end
end
