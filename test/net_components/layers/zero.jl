using Test

@testset "zero.jl" begin

    @testset "Zero" begin
        @testset "Base.show" begin
            p = Zero()
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "Zero()"
        end
	@testset "Initialize with tightening algorithm" begin
	    p = Zero([1,2,3])
	    @test p === 0
	end
    end

end
