@testset "transform" begin
    dims = (10,5)
    rng = MersenneTwister(123546)
    X = rand(rng, dims...)
    @testset "IdentityTransform" begin
        @test KernelFunctions.apply(IdentityTransform(),X)==X
    end
end
