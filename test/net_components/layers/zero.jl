using Test

@testset "zero.jl" begin

    @testset "Zero" begin
    	 @testset "Flatten" begin
	     p = Zero()
             @test p([1,2,3,4]) == 0
	 end
        @testset "Base.show" begin
            p = Zero()
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "Zero()"
        end
    end

end
