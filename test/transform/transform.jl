@testset "transform" begin
    dims = (10,5)
    rng = MersenneTwister(123546)
    X = rand(rng, dims...)
    @testset "IdentityTransform" begin
        @test KernelFunctions.apply(IdentityTransform(), X) == X
    end
    @testset "ColVecs" begin
        vX = KernelFunctions.ColVecs(X)
        t = ARDTransform(rand(dims[1]))
        @test KernelFunctions.apply(t, vX) ≈ KernelFunctions.ColVecs(KernelFunctions.apply(t, X, obsdim = 2))

        Y = rand(rng, reverse(dims)...)
        vY = KernelFunctions.ColVecs(Y')
        t = ARDTransform(rand(dims[1]))
        @test KernelFunctions.apply(t, vY) ≈ KernelFunctions.ColVecs(KernelFunctions.apply(t, Y, obsdim = 1)')
    end
end
